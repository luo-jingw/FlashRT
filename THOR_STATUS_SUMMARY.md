# ImageWAM on FlashRT — Thor 部署状态总结

测试口径（除特别说明外）：Jetson AGX Thor，MAXN、GPU 锁频。服务并测量两个工作负载——LIBERO 配置：双相机 224×448（14×28 patch 网格）、512 个补齐文本 token、8 维 proprio、10 步去噪、horizon=64、指令 16–31 个有效 token；目标配置：3 个 256×256 视角、指令 16–128 个 token 加 128 token 的补齐缓冲、horizon=32、10 步去噪、8 维 proprio。两者用同一个入口测量：`benchmarks/imagewam_thor_path_bench.py --workload libero|target`。真实 checkpoint；`infer()` 含 VAE 与 proprio，不含 Qwen3 文本编码。

## Pipeline 架构

一个 CUDA Graph 捕获的推理管线，用 C++/CUDA kernel + cuBLASLt/CUTLASS GEMM 通过裸指针调度，不走 PyTorch eager：

- **输入**：Qwen3-4B 文本编码（图外，每个 prompt 一次）；FLUX.2 VAE 图像编码（可选原生 NHWC 编码器，可进图）；proprio 状态（线性投影后插入文本 context，位置与官方一致）。
- **Backbone**：FLUX.2 DiT，5 层 double-stream + 20 层 single-stream，per-head QK-Norm、4 轴 RoPE、AdaLN modulation。
- **ActionDiT**：5 double + 20 single，10 步 flow-matching 去噪，官方 shift-based schedule。
- **输出**：反归一化到真实 action space。
- **执行**：prefill + 10 步 denoise 捕成一张 CUDA Graph，`infer()` 只是 replay；启用 `text_trim` 时每个有效文本长度一张图。
- **对外接口**：部署入口 `load_imagewam(ckpt_path, workload, profile=..., precision=..., calibration_path=...)`；Python `infer()`；`frt_model_runtime_v1` ABI（`io="python"`）；原生 C++ overlay（`io="native"`，热路径不经过 Python）。

## 部署配置：workload / profile / precision

解析链是 `workload + structure + profile + precision + calibration_path` → `resolve_config(...)` → `(dims, options)` → `from_config` → frontend；解析在构造后不再改变。

- **workload**：服务的工作负载，由 `ImageWAMWorkload` 描述。部署方给出相机数、每视角图像尺寸、文本长度、action horizon、动作维度、proprio 维度、去噪步数、调度 shift；序列布局由它派生：`x0`、`img_len`、`a0`、`total`、`ref_h`、`ref_w`、`dt`，以及原生 VAE 进图时的 `vae_graph_input`。派生值互相矛盾时在解析阶段报错，不再手填这些整数。LIBERO 工作负载是 `ImageWAMWorkload.libero()`：两个 224×224 视角、512 token 文本、horizon 64、7 维动作、8 维 proprio、10 步去噪、shift=5.0，派生 `x0=513`、`img_len=392`、`a0=905`、`total=969`、`ref_h×ref_w=14×28`。
- **profile**：一组开关的具名集合，按名字选用，代表**服务配置**（构造函数保留自己的历史默认：不裁文本）。`default` 是服务默认：nvfp4、**裁文本**、backbone 位点的 FA4 按机器自动解析（`use_fa4=None`：compute capability 11.x 且 FA4 runtime 可导入就用 FA4，否则走 cuBLAS 链；`FLASHRT_THOR_FA4=0` 强制走链，显式 `use_fa4` 优先）、torch VAE 编码器在图外、无 AWQ。`fast` 在 `default` 之上再加 FA4 的 mot 位点与原生 VAE 进图（opt-in：FA4 首次调用要编译且可能回退，进图 VAE 改的是图结构）。`native` 是 native 消费方的具名集合：**同样裁文本**，与 `default` 只差 FA4——两个位点都显式关，因为 native C++ pipeline 没有 FA4 注意力。
- **precision**：覆盖 profile 的精度档位。
- **calibration_path**：静态 FP8 与 AWQ 所需的校准文件。

`resolve_config` 是唯一的合法性判定点，非法组合抛一条带规则编号（R1–R11，以及值域规则 V1）的错误。日志里的 `effective_config` 行由 `config_resolver.format_effective_config` 产出，比较脚本、矩阵脚本与 runtime 身份打印同一个字符串，因此 profile 与精度是一份可记录的部署身份。

同一组按长度捕获的图也由 native 侧承载：native model runtime（`io="native"`）按长度 adopt 前端的每张图（`use_graph(key, exec)`，key 就是 `x0`），服务时用 `set_text_length(key)` 在热路径上选长度；native **pipeline** 自己也按文本长度记录：`pipeline_resources()` 描述**当前**长度（序列维度与 backbone RoPE 表跟着当前长度，缓冲区仍是按最长长度分配的那一套），`set_pipeline` 安装它那张表携带的 key 的 pipeline 并把它选为当前长度——只替换该 key 的 pipeline 与该 key 的图，其它 key 的 pipeline 与图都保留；`capture_pipeline_text_lengths` 把前端已捕获的每个长度装上并各记一张图，最后恢复原来的长度。所以 `text_trim` 对 native 路径与另外两条一样可用，对 native 消费方特有的只剩规则 R6。这条路径不改动未裁剪的一 key 路径：节点数不变（native 5324 / Python 5348），native-vs-Python graph 的 `array_equal` 行仍全绿；schema 记录也不变（Python 声明与 C++ native verbs 各 7 条记录，与 golden 逐字节相同）。

同一 workload 也是 runtime 与校准文件身份的一部分：ABI 描述与 `setup_identity` 在 `dims.<key>` 之外带 `workload.<field>`（`num_views`、`image_h`、`image_w`、`text_max_len`、`action_horizon`、`action_dim`、`proprio_dim`、`num_steps`、`shift`）；这些条目是附加描述，已记录的校准文件仍然有效。

新入口在 Thor 上验证过（`c20f3a0`）：同一配置下运行时打印的 `effective_config` 与 `config_resolver` 逐字符一致（`default` 与 `fast` 两行都比对过）；导出的 runtime identity 带全部九个 `workload.<field>`，其 ABI 与 `infer()` bit-exact，native schema 与 golden 一致。

## 用了哪些优化

**1. 量化精度**
- 默认 **NVFP4**（Blackwell 原生 block-scaled 4-bit）：权重离线量化，激活逐次动态量化，backbone 与 ActionDiT 全部 GEMM 走量化路径。
- 可选：NVFP4 + AWQ（逐通道 scale 折进权重）；`e0m3_hadamard`（Hadamard 旋转 + E0M3 均匀 4-bit 网格，走同一条 SM100 block-scaled tensor core 路径）；`fp8_static` / `fp8_static_cutlass`（真实校准，64 帧 LIBERO、142 个校准点）；`fp16` 作精度基线。

**2. 算子融合**（以减少显存带宽与 kernel 数为目的）
- QKV 合并为一个 GEMM；single-stream 的 `linear1`（qkv + mlp gate/up）与 `linear2`（attn_out + mlp_down）各合并为一个 GEMM。
- AdaLN（norm + scale + shift）一个 kernel；门控残差与下一层 AdaLN 融合，含跨层边界，一次 pass 的 kernel 数由 7082 降到 4968。
- SiLU-GLU MLP 一个 kernel。
- VAE 预处理：256 项查找表的融合 kernel。

**3. 序列与注意力**
- `text_trim`：每个 prompt 只处理有效文本 token（x0 = n_valid + 1），backbone 行数由 905 降到约 420，结果等价于官方的 masked attention。
- FA4 注意力（backbone 与 mot 两个位点，失败时回退 cuBLAS 链路）。

**4. VAE**
- 原生 NHWC 编码器，GroupNorm+SiLU 融合，可捕进主图。

**5. 执行调度与工程**
- 整图 CUDA Graph；per-shape cuBLASLt 算法调优；可选 ActionDiT 小 M 的 CUTLASS tile 逐形状选择。
- 校准文件（记录 checkpoint、shape、`text_trim` 的身份，运行时校验）；LIBERO 精度/延迟 gate（fixture + 每设备 baseline）；Jetson 时钟状态记录。

## 可选项与当前默认

| 选项 | 当前默认 | 作用 | 限制 |
|---|---|---|---|
| `workload` | `ImageWAMWorkload.libero()` | 服务的工作负载；序列布局与 `vae_graph_input` 由它派生并校验 | 字段必须与 checkpoint 的结构一致（规则 R7） |
| `profile` | `default` | 开关的具名集合；`default` 含 `text_trim`，backbone 位点的 FA4 按机器自动解析；`fast` = 再加 FA4 的 mot 位点 + 原生 VAE 进图（93.2 ms 对 default 的约 115 ms，nvfp4，libero_spatial） |`native` = 同样裁文本 + FA4 两位点显式关（native pipeline 没有 FA4 注意力），其余与 `default` 相同；`fast` 的 FA4 首次调用编译、失败回退 |
| `precision` | `nvfp4` | 精度/速度档位 | `fp8_static*` 需要校准文件 |
| `text_trim` | 关 | 按有效文本长度裁剪；开启后每个有效文本长度一张图，Python `infer()`、ABI 与 native 三条路径都能服务（native 侧由 `capture_pipeline_text_lengths` 逐长度安装并捕获） | — |
| `text_trim_cache_size` | 32 | `text_trim` 预捕获图的张数上限，超出按 LRU 淘汰 | 显存只在首次捕获付出：首图 +218.0 MiB reserved / +206.3 MiB allocated，其后每张 +0.0 / +0.1 MiB，默认上限 32 的总代价在 221 MiB 量级（`a84916a`，nvfp4，15 个 LIBERO 长度） |
| FA4（`FLASHRT_THOR_FA4`、`use_fa4_mot`） | backbone 位点：机器能跑就开（`FLASHRT_THOR_FA4=0` 强制走 cuBLAS 链）；mot 位点：关 | 注意力 kernel | 首次调用编译，失败自动回退 |
| `vae_encoder="native"` / `vae_graph_input` | 关（torch 编码器） | 原生 VAE / 进图 | — |
| `nvfp4_awq` | 关 | NVFP4 精度补偿 | 需要校准文件；原生 runtime 不支持 |
| `gemm_variant_autotune` | 关 | ActionDiT tile 逐形状选择 | 仅 NVFP4/FP8 CUTLASS 档位 |
| 融合项 | 开 | 见上 | `fp16_cutlass` 档位不做 `linear2` 合并 |

## Thor 实测结果

### 逐项收益（nvfp4，LIBERO spatial，`infer()` P50）

| 配置 | P50 | vs official cosine（median） |
|---|---:|---:|
| 当前默认 | 约 202–203 ms（gate 203.3） | 0.9976 |
| 只开 `text_trim` | 115.2 ms（libero_10 115.0，goal 129.9） | 0.99936 |
| 只开原生 VAE 进图 | 190.9 ms | — |
| 只开 FA4 backbone | 174.6 ms | 0.99751 |
| `text_trim` + FA4 双位点 + 原生 VAE 进图 | **106.1 ms** | **0.99933**（min 0.99887） |

三项叠加明显小于各项单独收益之和（−96 ms 对 −125 ms），`text_trim` 已经去掉了大部分 padding 上的注意力开销。表中叠满一行的 106.1 ms 属于 `c20f3a0` 一轮，见下。

### `eccf14f` 轮：两个工作负载与三条服务路径（nvfp4，`infer()` P50）

一次完整验证轮，commit `eccf14f`：Jetson AGX Thor、MAXN、GPC 1.575 GHz、`emc_locked=null`、GPU 独占；原始日志在 `/home/jingwu/thor_val/0919s/`。下面的数字都是端到端 `infer()` P50（ms），「vs official」是与官方模型的动作 cosine 中位数。

LIBERO 配置（`--workload libero`）：

| 任务 | 配置 | P50 | vs official（median） |
|---|---|---:|---:|
| libero_spatial | `default` | 202.4 / 202.1 / 202.0（三次重复） | — |
| libero_spatial | `stack` | 93.1 / 92.8 / 93.3（三次重复） | 0.99931–0.99936 |
| libero_goal | `default` | 203.0 | 0.99558 |
| libero_goal | `vae_trim` | 118.7 | 0.99937 |
| libero_goal | `stack` | 92.6 | 0.99930 |
| libero_goal | `profile=fast` | 92.7 | — |
| libero_10 | `default` | 202.0 | 0.99765 |
| libero_10 | `vae_trim` | 103.0 | 0.99926 |
| libero_10 | `stack` | 93.5 | 0.99925 |
| libero_10 | `profile=fast` | 93.7 | — |

Gate（nvfp4）：202.2 ms，通过。同轮同一会话内 `default` 连续三次重复的极差 0.4 ms、`stack` 0.5 ms；门禁、矩阵与路径基准的每一行都没有回退 FA4（`FA4 fallback=None`）。

同一组开关（`stack` / `profile=fast`）在本轮是 92.6–93.7 ms，`c20f3a0` 一轮是 106.1 / 106.8 ms；两轮的机器状态不同——同配置 `default` 在 `c20f3a0` 一轮是 225.2–225.5 ms，本轮是 202.0–202.4 ms。

叠满开关（`text_trim` + FA4 双位点 + 原生 VAE 进图）下换精度，libero_spatial 同一会话的 `default` / `stack` 两行：

| 精度 | P50（`default` / `stack`） | vs official（`default` / `stack`） |
|---|---|---|
| `e0m3_hadamard` | 198.9 / 90.5 | 0.99786 / 0.99968 |
| `fp8_static_cutlass` | 219.5 / 104.6 | — |
| `fp16` | 273.7 / 223.5 | — |

fp16 的链式复核：`default` 306.6 ms → `vae_trim` 234.9 ms → `stack` 222.9 ms。

AWQ（nvfp4 + AWQ）：动作 cosine 对 fp16 为 0.99970，未加 AWQ 的 nvfp4 为 0.99939；P50 202.2–202.9 ms，没有增加。真实 FP8 校准：对 fp16 的 cosine 0.99997、MAE 比 1.000；占位校准约 0.90、MAE 比 1.77。

三条服务路径（同一进程，LIBERO）：`default` 的 `infer()` 202.3 ms、ABI 184.2 ms、native 183.8 ms；`profile=fast` 在文本长度已预捕获时 `infer()` 93.2 ms、ABI 95.1 ms，native 该轮跳过（当时 native 面不接受裁剪）。

`text_trim` 的捕获开销：已预捕获的文本长度切换一张图用 0.000–0.012 s，首次使用时才捕获的长度用 0.42–0.58 s。

### `a84916a` 轮：目标工作负载可服务、LIBERO 回归与精度 gate（nvfp4，`infer()` P50）

commit `a84916a`：Jetson AGX Thor、MAXN、GPC 1.575 GHz、`emc_locked=null`、GPU 独占；原始日志在 `/home/jingwu/thor_val/0919e/`。本节数字都是端到端 `infer()` P50（ms），「vs official」是与官方模型的动作 cosine 中位数。

目标配置（`--workload target`，3 × 256×256、horizon 32）现在是受支持的配置：VAE 编码几何按工作负载的每视角尺寸走，`view_shape=(3,256,256)` 与 `img_len=768`（16×48）在同一路径上一致成立，`infer()` 跑通。同一进程三条服务路径：

| 配置 | `infer()` | ABI | native |
|---|---:|---:|---:|
| `default` | 216.93 | 173.55 | 173.27 |
| `fast`（预捕获） | 137.64 | 139.35 | 跳过（当时 native 面不接受裁剪） |

`fast` 相对 `default` 省 79.3 ms；文本长度已预捕获，因此 `fast` 的行不含首次捕获开销。

同一目标配置在 `text_trim` 下按有效文本 token 数取三个点（同一进程，每个长度一张采纳图）：

| 有效文本 token | `infer()` | ABI | native |
|---|---:|---:|---:|
| 16 | 197.00 | 153.51 | 跳过（当时 native 面不接受裁剪） |
| 72 | 207.06 | 161.68 | 跳过（当时 native 面不接受裁剪） |
| 128 | 217.50 | 172.05 | 跳过（当时 native 面不接受裁剪） |

ABI 面能服务裁剪后的 prompt（每个长度一张采纳图），native 面在该轮还不能（当时 native 面不接受裁剪）。

上一轮（`eccf14f`）测得的布局与两行仍成立：viewport 正确报出 `view_shape=(3,256,256)`，ABI 152.7 ms、native 154.5 ms（图像 token 用占位值）。

LIBERO 回归：VAE 几何改动是保真中性的——`libero_spatial` nvfp4 `default` 复现上一轮的数值完全一致（vs official min 0.99418 / median 0.99764、MAE 0.18290），P50 202.6 ms，对上一轮的 202.0–202.4 ms。

精度 gate：此前被阻塞的两个 gate 现在都能跑且通过——`e0m3_hadamard` vs official 0.99781 / min 0.99434、P50 222.2 ms（该精度没有 Thor 延迟 baseline，延迟部分不设门禁）；`fp8_static_cutlass`（真实校准）vs official 0.99830 / min 0.99557、P50 233.5 ms（同样不设门禁）。

`text_trim` 的显存代价：在 Thor 上实测（nvfp4、FA4 关、15 个 LIBERO 长度），第一张捕获图付出 +218.0 MiB reserved / +206.3 MiB allocated，其后每张 +0.0 / +0.1 MiB——后续图共用同一个捕获池。因此默认 `text_trim_cache_size=32` 的总代价在 221 MiB 量级，不是 32 倍的首图代价。

淘汰的代价（缓存上限 2、不预捕获）：重新访问一个已被淘汰的长度时 `set_prompt` 用 0.636 / 0.503 / 0.468 / 0.465 s；已预捕获的情形是 0.012 / 0.000 / 0.000 s。

### `0920s4` 轮：native model runtime 每个文本长度一张采纳图（nvfp4）

commit `a4852be`：Jetson AGX Thor、MAXN、GPC 1.575 GHz、`emc_locked=null`、GPU 独占；原始日志在 `/home/jingwu/thor_val/0920s4/`。本轮只重建了 native C++（`c_api.cpp`、`native_runtime.cpp`、`native_pipeline.cpp`、`fp4_linear.cpp`），kernel 与 fp4 目标本来就是当前的，ctypes 布局检查静默通过。延迟是每路径一个墙钟计时器的 P50（ms）。

`IMAGEWAM_NATIVE_PRECISION=nvfp4 pytest tests/test_imagewam_native_pipeline.py tests/test_imagewam_native_runtime.py -q`：38 passed。两个长度的 tick（`x0=6` 先 tick，再 `14`）的 `actions` 与 `actions_raw` 对 `infer()` 都是 `array_equal=True`、`max_abs=0`；native manifest 记 `text_lengths={'default_key': 14, 'keys': [6, 14], 'per_prompt_length': True}`；对没有采纳图的长度 `set_text_length(14)` 返回 `rc=-2`。未裁剪的一 key 路径（`tests/test_imagewam_native_pipeline.py`）整文件过。

| gate | 结果 |
|---|---|
| `tests/gate_imagewam_native_schema_parity.py --precision nvfp4` | PASS；Python 声明与 C++ native verbs 各 7 条记录，与 golden 逐字节相同（`x0` 不在记录里，没有重新基线化） |
| `tests/gate_imagewam_native_parity.py --precision nvfp4 --graph python --bench-iters 50` | 每个 tick 行 `array_equal=True`；两个调用 mutant 全检出；P50 python 207.53、native 207.13 |
| `tests/gate_imagewam_native_parity.py --precision nvfp4 --graph native --bench-iters 50` | 每个 tick 行绿；六个 mutant 全检出；P50 python tick 206.21、native tick 204.33，graph replay python 205.74、native 204.18 |

节点数 native 5324 / Python 5348，与上一轮 `08_gate_native.log` 相同。这轮的 P50 比那一轮 `08_gate_native` 的约 182 ms 高约 22 ms：本轮测到的是节点数不变、未裁剪路径的数值不变，没有测同一二进制在两轮机器状态下的 A/B，所以这 22 ms 记作会话差异，而不是实测到的回退。

该轮 native **pipeline** 自己的捕获还是单长度，所以这些长度上的 native 服务来自 model runtime 的采纳路径（当时 `consumer="native"` 的 `text_trim` 不被接受；native pipeline 现在自己按文本长度安装并捕获，见「Pipeline 架构」一节）。

### 新入口下的同一会话阶梯（`c20f3a0`，libero_spatial，nvfp4，`infer()` P50）

一次矩阵运行，`N_TASKS=10 FRAMES=0,60 SEEDS=0,1`，真实 checkpoint；`profile_*` 两行经 `load_imagewam` 构建，开关行经同一入口加对应 expert 覆盖。`fast` 与 `stack` 是同一组开关。

| 行 | P50 | 边际 | vs official（median） |
|---|---:|---:|---:|
| `default` | 225.5 | — | — |
| `vae` | 190.1 | −35.4 | — |
| `vae_trim` | 102.9 | **−87.2** | — |
| `vae_trim_fa4bb` | 99.0 | −3.9 | — |
| `stack` | **93.2** | −5.8 | 0.99934 |
| `stack_no_vae` | 104.8 | 相对 `stack` +11.6 | — |
| `stack_no_trim` | 131.4 | 相对 `stack` +38.2 | — |
| `profile=fast`（同 `stack` 开关） | 106.8 | — | 0.99936 |
| `profile=default`（同 `default` 开关） | 225.2 | — | 0.99764 |

FA4 两个位点都不回退（`FA4 fallback` 为 `None`）。按 C 节的判据，`stack` 相对 `vae_trim` 低 9.7 ms 且 vs official 不劣，达到「FA4 转默认」的工作门槛；原生 VAE 同理。默认值未改。

读这张表的两个口径问题（ISSUE-082）：`default` 行的记录基线 202–203 ms 来自只跑 frontend 的 gate，与带官方对照的 e2e 不是同一口径，`c20f3a0` 没有同口径的 gate 数；同一组开关在同一会话里出现两次，93.2 与 106.8 ms 相差 13.6 ms，比 C 节用的 2 ms 工作阈值大。`eccf14f` 轮回答了这两点：同轮 gate 202.2 ms 与 e2e `default` 202.0–202.4 ms 同口径且一致，同一会话内 `default` 三次重复的极差 0.4 ms、`stack` 0.5 ms；106.1 / 106.8 ms 属于 `c20f3a0` 那一轮偏高的会话状态。

微基准（随机权重、不含 VAE）：融合项合计约 −9.7 ms；FA4 backbone −26.9 ms，backbone + mot −58.5 ms；VAE 编码 stage 19.4 → 8.3 ms（进图）；VAE 预处理 kernel 0.97 → 0.13 ms。

### 各精度（未叠加其他选项，同一次运行，fp16 参考 275.2 ms）

| 精度 | `infer()` P50 | vs official（median，LIBERO gate） | MAE vs GT |
|---|---:|---:|---:|
| fp16 | 275.2 ms | 0.99835 | 0.18363 |
| fp8_static（真实校准） | 228.0 ms | 0.99829 | 0.18366 |
| fp8_static_cutlass（真实校准） | 220.3 ms | — | — |
| **nvfp4** | **202.3 ms** | 0.99744 | 0.18526 |
| nvfp4 + AWQ | 202.3–202.7 ms | — | — |
| e0m3_hadamard | 198.9 ms（同次运行的 nvfp4 为 202.3） | — | — |

### 叠满配置下的精度对比（`text_trim` + FA4 双位点 + 原生 VAE 进图，libero_spatial，10 task × frame 0/60 × seed 0/1）

| 精度 | vs official median (min) | MAE vs GT median | P50 |
|---|---|---:|---:|
| **nvfp4** | 0.99933 (0.99887) | 0.186 | **106.1 ms**（`c20f3a0`） |
| fp8_static_cutlass（trim 校准） | 0.99995 (0.99990) | 0.186 | 115.3 ms |

官方参考 MAE 中位数 0.1855。

### FlashRT vs 官方 PyTorch 实现

官方为 bf16 eager（`nn.Linear`→cuBLASLt + SDPA），同输入配置，在另一次会话中测得。

| 路径 | P50 | 相对 FlashRT nvfp4 默认 |
|---|---:|---:|
| 官方 bf16 eager，端到端 | 453.6 ms | 2.2× |
| FlashRT nvfp4 默认 | 203.3 ms | 1.00× |
| FlashRT nvfp4 叠满 | 106.1 ms（`c20f3a0`） | 0.52×（快 1.9×；相对官方约 4.3×） |

官方侧 `torch.compile`：`inductor` 在 Thor 上无法编译；`cudagraphs` 比 eager 更慢（514.4 ms）。

### INT8 / INT4（SM80 CUTLASS）

Thor（Blackwell 第五代 Tensor Core，`tcgen05.mma`）原生支持 TF32/FP16/BF16/INT8（legacy）与 MXFP4/NVFP4/MXFP6/MXFP8（block-scaled 浮点），没有整数 INT4 通路；Ampere/Ada 的 `s4×s4→s32` MMA 在 Thor 上走兼容慢路径（INT4 prefill 1253.6 ms，INT8 138.7 ms）。Thor 的原生 4-bit 方案是 NVFP4。

## 与 Pi0.5 部署方案的对照

| 技术 | Pi0.5 | ImageWAM 现状 |
|---|---|---|
| 通用 ABI（`frt_model_runtime_v1`） | Python + 原生 C++ | 已接入（`io="python"`、`io="native"`），与 `infer()` bit-exact |
| 真实数据校准 | 8 帧 LIBERO，percentile 99.9，不可变 artifact | 64 帧 LIBERO，artifact 记录身份并在打开时校验 |
| AWQ | 逐通道 scale 折进 NVFP4 权重 | 已实现，可选 |
| Hadamard 旋转 INT4（E0M3） | 有 | 已实现（`e0m3_hadamard`），可选 |
| 门控残差 + 下一层 Norm 合并 | 一个 elementwise kernel | 已实现，含跨层 |
| 图像归一化查找表 | 256 项 FP16 LUT | 已实现 |
| 小 M CUTLASS tile | 有 | 已实现，可选（实测收益 ≤0.9%） |
| 精度/延迟回归门禁 | 有（时钟锁定检查、同批次 A/B、per-view baseline） | LIBERO gate + 每设备 baseline + 时钟状态记录 |
| 多子图 stage 拆分 | RTC 前缀复用 / VJP 引导 | 不需要 |
| 完整 attention 链路融合 | 已测试，慢 5–7 倍，放弃 | 改用 FA4 |

Pi0.5 自身的 Thor 数字（NVFP4 + FA4，完整 `infer()`）：1 / 2 / 3 相机 23.01 / 27.17 / 31.74 ms，是不同的模型，仅作量级参考。
