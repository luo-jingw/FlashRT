# 推理优化方法论（π0.5、ImageWAM）

FlashRT 如何把一个机器人策略从参考实现做成快速的服务运行时，整理成一套按顺序执行的流程。每一步都注明在 π0.5 或 ImageWAM 上的应用和实测结果。顺序很重要：每个阶段都以前一个阶段成立为前提。英文版见 [`inference_optimization_methodology.md`](inference_optimization_methodology.md)。

| 阶段 | 回答的问题 | 完成标准 |
|---|---|---|
| 0. 参考与门禁 | 什么算"正确"？ | 有固定输入集、参考输出和数值门禁 |
| 1. 静态执行 | 整个推理能否回放？ | 一个（或少数几个）CUDA graph，内部没有 host 端工作 |
| 2. 测量 | 时间花在哪，为什么？ | 模块 × 算子 × 精度的表格，每格标出瓶颈 |
| 3. 去掉多余计算 | 哪些计算根本不需要？ | 去掉死计算和重复计算，并检查过算法层面的节省 |
| 4. 融合 | 哪些 kernel 受启动次数或数据量限制？ | 融合 kernel 在每台目标设备上都做过 A/B |
| 5. 精度 | 每台设备上哪一档最快且精度可接受？ | 按设备选定精度档，校准和门禁都通过 |
| 6. GEMM 与 attention 后端 | 每个形状是否都用上了最合适的 kernel？ | 在真实 M 下逐形状测量并选定 kernel |
| 7. 任务级验证 | 策略还能成功吗？ | 闭环成功率在置信区间内与参考一致 |

## 阶段 0：参考、固定输入集与门禁

- 先跑一次官方实现，保存它的输出作为参考。对照的对象是 FP32（或官方 BF16）参考，不能是运行时自己的输出（`docs/calibration.md` §10）。
- 固定输入：做一个带版本号的输入集，包含真实图像帧、prompt、proprio 和初始噪声（`flash_rt/datasets/imagewam_gate_fixture.py`、`tests/gate_imagewam_libero.py`；π0.5 的 FP4 测试框架用 8 个观测，并固定噪声 seed，见 `docs/pi05_thor_decoder_fp4_e2e.md`）。
- 本仓库使用的门禁：
  - π0.5 接入阶段：action 余弦相似度 ≥ 0.995；部署阈值 0.998（`docs/adding_new_model.md` §3、`docs/calibration.md` §10）。
  - π0.5 FP4：整体 action 余弦 ≥ 0.999，每个样本 ≥ 0.995（`docs/pi05_thor_decoder_fp4_e2e.md`）。
  - 不应改变数值的重构必须逐位一致（`torch.equal`），例如 exec contract（`docs/exec_contract.md` §8.2）、原生 C++ 实现（`docs/pi05_native_cpp.md`），以及 ImageWAM 的 `fuse_res_norm`、`qkv_split_norm_rope_fp16` 等融合。
- 逐位一致的对照对象是**当前默认路径**，不是旧路径。`qkv_split_norm_rope_fp16` 最初只和旧的 `rms_norm_fp16` 逐位一致；`vec_rms_norm` 成为默认后，它按新的归约顺序重写（`opportunities.md` OPT-032）。

## 阶段 1：静态执行

目标：整个 `infer()` 只剩输入准备加一次 graph 回放。

- 把整个调用捕获成 CUDA graph。
  - π0.5 捕获 SigLIP、encoder 和 10 步 decoder，约 2,840 个节点，TensorRT / Myelin 引擎约 21,000 个（`docs/optimization-details.md` §1.6）。
  - ImageWAM 捕获 prefill 和 10 步去噪循环；VAE 也可以放进 graph（`vae_graph_input`，OPT-021）。
- 捕获规则（`docs/kernel_fusion.md` §6、`docs/adding_new_model.md` §2）：
  - 只传指针和整数；
  - 所有缓冲区在加载时分配；
  - 捕获区域内不能有内存分配、同步、`.cpu()` 或依赖数据的 Python 分支；
  - warmup 形状必须等于服务形状；
  - 标量时间步要在设备上构造（`docs/pi05_scalar_capture.md`）。
- 和观测无关的量全部预计算：
  - 每步的 AdaRMSNorm / AdaLN 调制量和时间嵌入在 `set_prompt` 时算好。π0.5 因此从 graph 中去掉了 370 个 dense GEMM（`docs/optimization-details.md` §1.4）；ImageWAM 按步预计算 `action_mods` / `head_mods`。
  - 文本编码每个 prompt 只做一次。
- 变长输入有两种做法：
  - 每个确切长度一个 graph 并缓存：π0.5 `state_prompt_mode="exact"`，ImageWAM `text_trim=True`（OPT-030）；
  - 一个最大长度的 graph 加 mask：π0.5 `state_prompt_mode="fixed"`，用 FA2 的 `seqused`。
  - 已知长度要提前捕获，服务过程中就不必再捕获（`precapture_text_lengths`）。
- 阶段之间零拷贝交接：上游直接写下游读取的缓冲区（`docs/exec_contract.md` §5b）。
- Thor：graph 实例化的耗时会有约 ±2 ms 的波动；π0.5 会反复重新捕获，直到得到一个快的调度（`docs/optimization-details.md` §1.6）。

## 阶段 2：先测量，再优化

- 建一个网格：模型模块 × 算子类别 × 精度 × 设备。ImageWAM 用 `benchmarks/imagewam_module_operator_profile.py` 来做（`docs/imagewam_rtx5090_profile.md`、`docs/imagewam_thor_profile.md`、`docs/imagewam_profile_compare.html`）。
- 交叉核对归因：
  - graph 回放的 kernel 时间应与 `infer()` 墙钟时间基本相等（ImageWAM 5090 上相差 0.2–1.8 ms）；
  - 各次 profile 迭代的 kernel 数必须相同。先跑一次丢弃用的 profiler 会话，可以避免 CUPTI 丢记录。
- 在同一设备上测出上限（各精度的 dense GEMM 峰值、DRAM 带宽），再给每格判断瓶颈：

  | 瓶颈 | 特征 | ImageWAM 示例 |
  |---|---|---|
  | 算力 | GEMM 接近实测峰值 | 5090 主干单流 GEMM，fp8 在 M = 416 时约达峰值的 100% |
  | 读权重带宽 | GEMM 时间 ≈ 权重字节数 / DRAM 带宽 | 5090 ActionDiT GEMM，M = 64，fp16 达 DRAM 带宽的 75–84% |
  | kernel 效率 | 离两个上限都很远 | 5090 NVFP4 ActionDiT GEMM，只用到 19–25% 带宽 |
  | 启动 / 延迟 | 大量 1–3 µs 的 kernel | 5090 ActionDiT 的非 GEMM kernel，每次推理约 1,800 个 |
  | 小 kernel 的数据量 | 逐元素 kernel 时间 ≈ 字节数 / 可达带宽 | Thor 上 `silu_glu_merged` 5.7 ms，是 5090 的 10 倍 |

- 每台设备的结论要分开记录：同一个 kernel 在 RTX 5090 上可能受启动次数限制，在 Thor 上却受带宽限制。

## 阶段 3：去掉多余计算

算法和结构层面的删减，通常比调 kernel 收益更大。

- **共享前缀只算一次**，让迭代部分读取它的 KV。ImageWAM 的动作专家读取主干逐层的前缀 KV；π0.5 的 decoder 读取 encoder 的 K/V 缓存。
- **去掉没人读的计算**：ImageWAM 主干最后一层只需要 K/V（`docs/imagewam_last_block_kv_only.md`）；文本裁剪到有效长度（OPT-030）。
- **先量化成本随循环次数和 token 数的变化**，再动 kernel。见 `docs/imagewam_rtx5090_profile.md`：去噪循环耗时随步数线性变化；减少图像 token 只缩短 prefill。
- **先检查迭代采样器的结构**，不要默认每一步都必要。
  - 在 ImageWAM LIBERO checkpoint 上，速度场的雅可比 ≈ (1/t)·I（`benchmarks/imagewam_velocity_jacobian_study.py`），预测 x̂₀ = x − t·v 几乎不依赖 x。
  - 因此只在 10 步网格的 {0, 9} 两个点上求值动作专家，和 10 步结果的 action MAE 只差 0.0006（`benchmarks/imagewam_few_step_study.py`）。
  - 闭环：LIBERO-10 上两者都是 48/50（fp8，50 个 episode）；RoboTwin 2.0（官方 torch policy，50 个任务 × 10 个 episode，`demo_clean`）在两种配置都评测过的 446 个 seed 上为 94.2% 对 94.6%，McNemar p = 0.83。
- **时间维度并行采样**（Picard / ParaDiGMS）是用更宽的 batch 换更少的串行步。只有当 M = 64·k 的批量单步远小于 k 个串行步时才划算。见 `docs/imagewam_rtx5090_batched_m_step.md`：RTX 5090 上 k = 10 时是串行一步的 2.8–4.6 倍。

## 阶段 4：融合

- 只针对阶段 2 标出的启动受限或数据量受限区域。
- 本仓库用过的融合：
  - **norm + 量化、残差 + norm + 量化**：π0.5 的 `rms_norm_fp8_static`、`residual_add_rms_norm_fp8_noweight_fp16`、`gate_res_adarms_fp8_static`；ImageWAM 的 `gate_res_ada_layer_norm_fp16`、`fuse_res_norm`（OPT-017）。
  - **Q/K/V 切分 + 逐头 norm + RoPE**：π0.5 的 `qkv_split_rope_kvcache_fp16`；ImageWAM 的 `qkv_split_norm_rope_fp16`（OPT-032）。
  - **门控 MLP 激活**：`gate_geglu_merged_fp8_fp16`、`silu_glu_merged_fp16`。
  - **合并 GEMM**：ImageWAM 单流层的 `linear1` 合并 qkv 和 mlp_in，`linear2` 也做了合并（OPT-015/016）。
  - **加载时把 norm 权重折进下一个 GEMM**：π0.5 的 `(1+w)` 折叠，无权重的 norm 约快 15%（`docs/kernel_fusion.md`）。
  - **向量化读取和打包写入**：每个 kernel 快 1.4–2.3 倍（`docs/optimization-details.md`）。
- **融合必须在每台目标设备上实测。** `qkv_split_norm_rope_fp16` 的两个版本：
  - 每个 (row, head) 一个 256 线程 block 的版本：在 RTX 5090 上省 1.0–1.7 ms，在 Thor 上慢 7.0–9.6 ms；
  - 改成每个 (row, head) 一个 warp 后：5090 上省 2.0–2.7 ms，Thor 上省 7.3–8.3 ms（OPT-032）。
  - 结论：启动次数不能预测结果，决定结果的是融合 kernel 本身的效率。
- **改变计算图拓扑的融合可能变慢。** π0.5 的 TensorRT 路径里：einsum→matmul 慢 5.18 ms，一个 RoPE plugin 慢 10 ms（`docs/kernel_fusion.md` §4–§5、`docs/optimization-details.md` §6）。

## 阶段 5：精度

- **每台设备的默认精度档由实测决定，不看位宽。** ImageWAM 最快档：RTX 5090 是 fp8（23.85 ms），Thor 是 nvfp4（80.22 ms）；RTX 5090 上 nvfp4（28.23 ms）反而比 fp8 慢。
- **FP8**：
  - 用校准得到的静态 per-tensor 激活 scale，去掉每次调用时的 absmax / scale kernel。π0.5 因此省掉约 630 个量化 kernel（`docs/optimization-details.md` §1.2）。
  - 用真实数据、多个样本、按分位数校准（π0.5：p99.9，RTX 上 N = 8，Thor 上 N = 64）。用真实数据重新校准后，LIBERO Spatial 从 91.8% 升到 98.2%（`docs/calibration.md`）。
  - alpha 用 float32 计算，权重 spec 的顺序要保持一致（`docs/calibration.md` §2.3）。
- **FP4（NVFP4）**：
  - 只在 kernel 能在服务形状上跑出吞吐时使用，精度不够时加 AWQ（`docs/imagewam_nvfp4_awq.md`；π0.5 encoder FFN 用 AWQ α = 0.8）。
  - 要把额外开销算进去：RTX 5090 上，ImageWAM ActionDiT 每步花 0.43 ms 在 fp16→bf16 转换、激活量化和 bf16→fp16 转换上，M = 640 时是 1.27 ms。
- **有些精度档会因精度问题被永久放弃**：π0.5 视觉部分用静态 INT8 后，余弦从 0.991 跌到 0.282。

## 阶段 6：GEMM 与 attention 后端

- **按服务时的 M 逐形状选 kernel。** 小 M 是策略模型的常见问题：action token 数在 ImageWAM 是 M = 64，π0.5 decoder 约 10。
- **调优前先弄清硬件和库对 kernel 的限制。**
  - CUTLASS 4.4.2 的 SM120 block-scaled（NVFP4）主循环要求 tile 的 M、N 都至少 128（受 scale factor TMA 限制），所以窄 tile（128×8 / 128×16）和 M = 64 的 tile 都编译不过；
  - 输出维度 1024 的形状于是只有 8 个 tile，而 GPU 有 170 个 SM；
  - 逐形状选择 cooperative / stream-K / split-K 后，ActionDiT 每步的 NVFP4 GEMM 从 955 µs 降到 780 µs（`docs/imagewam_rtx5090_batched_m_step.md`、OPT-038）；
  - stream-K / split-K 每次启动前都要把 workspace 清零，否则第二次启动会挂起。
- **单独测赢的 tile，放进流水线可能输。** π0.5 在 Thor 上的 cluster tile 反而慢 2.2–14 ms（`docs/pi05_thor_decoder_fp4_e2e.md`），必须做端到端确认。
- **autotune 要小心**：SM89 上一个坏的 FP8 NT 候选会污染 context，所以那里的 autotune 被关掉了（`pipeline_rtx.py`）。
- **attention 后端按设备和调用位置选**：π0.5 在 RTX 上用 FA2；Thor 上用 FA4（ImageWAM 的两个位置都用）；其余情况用 cuBLAS 的 QKᵀ / softmax / PV 三段式。要在真实序列长度下测量（OPT-019）。

## 阶段 7：任务级验证

- 任何改变数值的改动（精度档、减少求值次数、校准），都要做闭环验证，不能只看余弦。
- 不同配置要跑同一批 episode 和初始状态，做配对比较，报告置信区间，并写明 episode 数。
- RoboTwin 的评测 seed 要先通过一个有随机性的专家检查，所以两种配置只共享部分 seed。要分别给出共享 seed 上的比较和整体比较。

## 测试规范

- **计时**：warmup 之后用 CUDA event 取 ≥100 次的 P50，同时报告 P10 / P90。
  - RTX 5090 上时钟有爬升过程，warmup 太短会测偏高（`docs/exec_contract.md` §8.3）；
  - π0.5 在 Thor 上测 FP4 时用 300 次 warmup。
- **A/B**：在同一个进程里按 ABBA 顺序交替测，并复用自动调优选出的 GEMM 算法（`benchmarks/imagewam_fusion_ab.py`）。
  - Thor 上有相差约 3 ms 的两种时钟状态，跨进程波动最多 21 ms（`issues.md` ISSUE-095）；
  - 所以只在同一会话内比较，并在独立进程里重复。
- **比较不同精度档时，每个进程只测一种精度。** 每个数字都要记录 commit、驱动、时钟状态和 GPU 是否独占；测试前确认没有其他任务占用 GPU。
- **单独测 kernel 时要放在 CUDA graph 里计时**，否则对 10 µs 量级的 kernel，host 端的入队开销会占主导。

## 两个模型上踩过的坑

- **拿运行时自己的输出当参考，会掩盖 scale 错误。** 一个 scale 顺序错误让余弦降到 0.933；用 float64 算 alpha 降到 0.9878（`docs/calibration.md` §4）。tactic 噪声让余弦上下波动 ≤ 0.002；scale 错误则会一直偏低（§6.1）。
- **融合 kernel 如果沿用了低效结构，在带宽受限的设备上会变慢**（OPT-032）。
- **装进 venv 的库可能覆盖基础环境的 cuBLAS。** ImageWAM 的 fp8 GEMM 因此报 status 15（`PROJECT.md`）。
- **block-scaled 缓冲区的 padding 必须初始化为零**，π0.5 NVFP4 曾因此出现 NaN（`docs/pi05_thor_decoder_fp4_e2e.md`）。
- **"kernel 更少"和"精度更低"本身都不等于更快**，要由每台设备上实测的端到端时间来决定。
