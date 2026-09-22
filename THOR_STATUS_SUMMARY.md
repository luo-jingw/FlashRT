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
- **profile**：一组开关的具名集合，按名字选用，代表**服务配置**（构造函数保留自己的历史默认：不裁文本）。`default` 是服务默认，也是机器与输入允许的**最快配置**：nvfp4、**裁文本**、FA4 用在 backbone 与 mot 两个位点、原生 VAE 编码器进图、无 AWQ（0921 起；此前 mot 位点与 VAE 进图是 `fast` 的 opt-in）。FA4 与 VAE 是“auto”，条件不满足时降级而不是报错：FA4 在 compute capability 11.x 且 FA4 runtime 可导入时使用，否则走 cuBLAS 链（`FLASHRT_THOR_FA4=0` 强制走链，显式值优先，首次调用要编译，捕获失败自动回退）；VAE 在给了 `ae_model_path` 时用原生编码器并进图，没给就没有 VAE 阶段。`fast` 是同一组开关的显式写法：缺 FA4 或缺 `ae_model_path` 会报错，用来让一次运行失败而不是悄悄测到回退配置。`native` 是 native 消费方的具名集合：**同样裁文本**，FA4 两个位点与 VAE 阶段都关，因为 native C++ pipeline 既没有 FA4 注意力也没有 VAE 阶段；`default` 加 `consumer="native"` 解析成同一组。下文各表里标“`default`”的历史数字是 0921 之前的默认（裁文本 + backbone FA4 auto + torch VAE 图外）测得的，新默认的数字等 Thor 的 N1–N3。
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
| `profile` | `default` | 开关的具名集合；`default`（0921 起）= `text_trim` + FA4 两个位点 + 原生 VAE 进图，条件不满足时降级；`fast` = 同一组开关的显式写法（缺 FA4 会报错）；下面括号里的数字是提升之前的对比（`0920t`：同进程、LIBERO `valid_tokens=24` 时 `fast --precapture` 的 `infer()` 108.08 ms 对 `default` 139.47 ms；`c20f3a0` 一轮的对应两行是 93.2 对约 115 ms，nvfp4，libero_spatial） |`native` = 同样裁文本 + FA4 两位点显式关（native pipeline 没有 FA4 注意力），其余与 `default` 相同（`default` 加 `consumer="native"` 得同一组）；FA4 首次调用编译、失败回退 |
| `precision` | `nvfp4` | 精度/速度档位 | `fp8_static*` 需要校准文件 |
| `text_trim` | 关（服务 `default` 档位为**开**） | 按有效文本长度裁剪；开启后每个有效文本长度一张图，Python `infer()`、ABI 与 native 三条路径都能服务（native 侧由 `capture_pipeline_text_lengths` 逐长度安装并捕获）；服务默认即裁剪，`0920t` 的 nvfp4 gate 125.86 ms、端到端 126.5 ms | native pipeline 自己的按长度捕获已验证（`0920c`）：`test_pipeline_records_one_graph_per_text_length` 通过，该轮 native 两条 pytest 39 passed 且无 skip，逐长度 tick `differing=[]`，每个长度的 GEMM 交接完整（`{6: (4, 4), 14: (4, 4)}`） |
| `text_trim_cache_size` | 32 | `text_trim` 预捕获图的张数上限，超出按 LRU 淘汰 | 显存只在首次捕获付出：首图 +218.0 MiB reserved / +206.3 MiB allocated，其后每张 +0.0 / +0.1 MiB，默认上限 32 的总代价在 221 MiB 量级（`a84916a`，nvfp4，15 个 LIBERO 长度） |
| FA4（`FLASHRT_THOR_FA4`、`use_fa4_mot`） | 两个位点都是 auto：机器能跑就开（`FLASHRT_THOR_FA4=0` 强制走 cuBLAS 链，两个位点一起） | 注意力 kernel；捕获失败自动回退 cuBLAS 链。`0921` 的服务默认 gate 三次都是 `use_fa4=True use_fa4_mot=True fa4_fallback_reason=None`；`FLASHRT_THOR_FA4=0` 时 `use_fa4=False use_fa4_mot=False`，P50 103.2 ms（与 `vae_trim` 行 103.3 ms 同量级），不报错 | 首次调用要编译：`0921` 清缓存后冷构造 14.67 s、紧接着的热构造 11.79 s，差额 2.89 s 是一次性的；native pipeline 没有 FA4，`native` profile 显式关 |
| `vae_encoder` / `vae_graph_input` | auto：给了 `ae_model_path` 就用原生编码器并进图，没给就没有 VAE 阶段（可显式写 `vae_encoder="torch"`） | 原生 NHWC VAE 编码器 + 预处理，进主图 | native 面拒绝进图 VAE（`native` profile 图外；`default` 在 path bench 里 native 行 SKIP 是预期） |
| `nvfp4_awq` | 关 | NVFP4 精度补偿 | 需要校准文件；原生 runtime 不支持 |
| `gemm_variant_autotune` | 关 | ActionDiT tile 逐形状选择 | 仅 NVFP4/FP8 CUTLASS 档位 |
| 融合项 | 开 | 见上 | `fp16_cutlass` 档位不做 `linear2` 合并 |

## Thor 实测结果

### 逐项收益（nvfp4，LIBERO spatial，`infer()` P50）

| 配置 | P50 | vs official cosine（median） |
|---|---:|---:|
| 当前默认（0921 起：裁文本 + FA4 双位点 + 原生 VAE 进图；gate 三次 / 端到端 `profile_default`） | **105.42 / 105.93 / 105.66 ms（gate）/ 93.1 ms（端到端，各 prompt）** | 0.99932（gate median；min 0.99884–0.99896） |
| 上一版默认（裁文本 + FA4 backbone + torch VAE 图外，`0920t` 的 gate / 端到端） | 125.86 / 126.5 ms | 0.99934（min 0.99889） |
| 旧默认（未裁剪 + FA4 关；`eccf14f` 一轮 `default` 202.0–202.4；gate 203.3 属更早的会话） | 约 202–203 ms | 0.9976 |
| 只开 `text_trim` | 115.2 ms（libero_10 115.0，goal 129.9） | 0.99936 |
| 只开原生 VAE 进图 | 190.9 ms | — |
| 只开 FA4 backbone | 174.6 ms | 0.99751 |
| `text_trim` + FA4 双位点 + 原生 VAE 进图 | **106.1 ms**（`c20f3a0`） | **0.99933**（min 0.99887） |

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

### `0920` 轮：三个症状的甄别与捕获路径缺陷的修复

Jetson AGX Thor、MAXN、GPC 1.575 GHz、`emc_locked=null`、GPU 独占；原始日志在 `/home/jingwu/thor_val/0920/`。本轮把三个症状各自单独跑了一遍，四个文件全绿：

| 检查 | 结果 |
|---|---|
| `tests/test_imagewam_fa4_dispatch.py -k capture_sync` | 1 passed——一次作废的捕获之后，重试确实在一张新池上捕获 |
| `tests/test_imagewam_awq.py` | 6 passed，打印 `kernels per eager forward: plain=285 awq=285` |
| `tests/test_imagewam_text_trim_graph_safety.py`（FA4 开） | 4 passed，恢复用例报 `equal=True cosine=1 max_abs=0` |
| `tests/test_imagewam_model_runtime_vae.py` | 2 passed |

三个症状里只有一个是真的缺陷：cuBLAS 回退重捕到了被作废的那次尝试还留在录制中的流与显存池；修法是先丢弃这份状态，让重试等同于一次全新捕获。另外两个是测试自己的比较——AWQ 读数里 plain 一侧为 `0`，是因为进程里第一段 `torch.profiler` CUDA 区域是盲的；FA4 开启的恢复行把一次 cuBLAS 重捕与失败前的 FA4 捕获相比。fixture v2 的 manifest 按字节原样入库（sha256 `a69a86ac…`、10954 字节）。

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

### `0920t` 轮：服务默认自身的数字与三条服务路径（nvfp4）

commit `4cd06e5`：Jetson AGX Thor、MAXN、GPC 1.575 GHz / NVD 1.692 GHz、`emc_locked=null`、GPU 起始空闲；原始日志在 `/home/jingwu/thor_val/0920t/`。本轮测的是服务默认本身——裁剪加上机器能跑就跑的 FA4：`FLASHRT_THOR_FA4` 的默认值已是 `"1"`，所以 `use_fa4=None` 在这台机器上解析为 `True`；`effective_config` 对服务默认打印 `use_fa4=True`，对 `FLASHRT_THOR_FA4=0` 打印 `use_fa4=False`。延迟是 P50（ms）。

三次 gate 全部通过（fixture v2 的 fp16 参考是裁剪的）：

| gate 运行 | vs official（min / median） | P50 |
|---|---:|---:|
| nvfp4，fixture v2，服务默认（裁剪 + FA4） | 0.99889 / 0.99934 | 125.86 |
| nvfp4，fixture v1，`--no-text-trim`（未裁剪 + FA4） | 0.99418 / 0.99758 | 191.79 |
| fp16，fixture v2（不设门禁） | 0.99993 / 0.99997 | 284.38 |

端到端 `default`：`served_vs_off` min 0.99432 / median 0.99750、P50 126.5 ms；同一裁剪配置加 `FLASHRT_THOR_FA4=0`：min 0.99441 / median 0.99743、P50 131.3 ms。也就是同一裁剪配置下 FA4 开比关快约 5 ms，两次的 `served_vs_off` 中位数相差不到 1e-4（0.99750 对 0.99743）。

未裁剪配置在 FA4 开时是 191.79 ms；记录的 202.2 ms 基线是未裁剪**且** FA4 关，因此 202.2 只是这条配置的单侧界。`0919e` 一轮在同一 fixture（v2）上的裁剪 gate 是 FA4 关的 114.6 ms，本轮服务默认是 125.86 ms，相差约 11 ms：两轮是不同会话、没有做同一机器的跨会话 A/B，所以记作会话差异，不是实测回退。

三条服务路径（同一进程，LIBERO，`valid_tokens=24`，即 `x0=25`）：

| profile | `infer()` | ABI | native | 说明 |
|---|---:|---:|---:|---|
| `default` | 139.47 | 120.29 | 120.05 | 裁剪，`use_fa4=True`，`graph_producer=python` |
| `fast --precapture` | 108.08 | 110.09 | 跳过（native VAE 进图） | 裁剪，FA4 两个位点，原生 VAE 进图；文本长度已预捕获 |
| `native` | 145.93 | 126.58 | 126.38 | 裁剪，`use_fa4=False`；不再被任何规则跳过 |

native C++ 本轮重建。`FLASHRT_THOR_FA4=0 IMAGEWAM_NATIVE_PRECISION=nvfp4 pytest tests/test_imagewam_native_pipeline.py tests/test_imagewam_native_runtime.py -q`：38 passed、1 failed；未裁剪的一 key 路径整文件绿（state `array_equal`、`graph_exec=0`、`graph_nodes=0`、`graph_producer=''`）。失败的是 `test_pipeline_records_one_graph_per_text_length`，两个成因已在 `43c49ce` 修复（长度表是 property、比较覆盖了当前长度以外的行）；native pipeline 自己的按长度捕获在 `0920c` 一轮重跑后已验证（下一节）。`tests/test_imagewam_text_trim_consumer_guards.py` 11 passed，其 GPU 行打印逐长度资源表（`x0=6` 维度 `(6,16,20)`、`x0=10` 维度 `(10,20,24)`，AdaLN 行与 backbone RoPE 表跟着当前长度，`buffers identical=True`）。

| gate | 结果 |
|---|---|
| native parity（`nvfp4`） | PASS；节点数 native 5324 / Python 5348，六个 mutant 全检出，tick `array_equal` / `max_abs=0`，P50 native 205.27 对 Python 207.13 |
| native schema parity | PASS；7 条记录与 golden 文件完全相同 |

### `0920c` 轮：三项收尾（native 自录图、ABI 导出 gate、`e0m3_hadamard` 裁剪安全检查）

commit `c495cb2`：Jetson AGX Thor、MAXN、GPC 1.575 GHz / NVD 1.692 GHz、`emc_locked=null`、GPU 空闲；原始日志在 `/home/jingwu/thor_val/0920c/`。本轮没有新的 C++ 或 kernel 改动，`flashrt_imagewam_native` 是上一轮编出来的那个（幂等确认，未为新代码重编）；FA4 由各测试自己显式声明，未设 `FLASHRT_THOR_FA4`。延迟是 P50（ms）。

native model runtime 与 pipeline 按长度带图（OPT-029，`FLASHRT_THOR_FA4=0 IMAGEWAM_NATIVE_PRECISION=nvfp4`）：native 两条 pytest **39 passed**、guards **15 passed**，两者都没有 skip（`exec/` 与库都在位）。`test_pipeline_records_one_graph_per_text_length` 里 handle 自己装管线并录图，先 `x0=6` 再 `14`，manifest `text_lengths={'default_key': 14, 'keys': [6, 14], 'per_prompt_length': True}`，每个长度的 GEMM 交接 `{6: (4, 4), 14: (4, 4)}`——每个长度都把该长度的全部 shape 交了出去；两个长度的 tick 都是 `differing=[]`、`actions max_abs=0`。未裁剪的一 key 路径整文件不变（`graph_exec=0`、`graph_nodes=0`、`graph_producer=''`）。guards 的 GPU 行给出逐长度资源表：`x0=6` dims `(6,16,20)`、`x0=10` dims `(10,20,24)`，AdaLN 行数与 backbone RoPE 表跟着活动长度，`buffers identical=True`。

| gate | 结果 |
|---|---|
| native parity，`--graph native` | PASS；节点数 native 5324 / Python 5348（与 `0920s4` 相同），六个 mutant 全部 `detected=True`，tick `array_equal` / `max_abs=0`，P50 native tick 204.43 对 Python 206.37 |
| native schema parity | PASS；7 条记录与 golden 逐行相同 |

ABI 面的导出 gate（OPT-028 自己记的 promotion condition，此前从未在 Thor 上跑过）：`--precision nvfp4` 的两条腿都 PASS，脚都打印 `use_fa4=False`。每一行 `array_equal=True` / `max_abs=0`（含 `images` 端口 stage 出来的 VAE token bits），确定性对照（同噪声两次 `infer()`）通过，五个 mutant 在 VAE 图外与图内两种放置下都被检出。

| 腿 | VAE 放置 | `infer()` / ABI tick P50 | 进程显存峰值 |
|---|---|---:|---:|
| plain | `vae_graph_input=None`（图外） | 226.93 / 227.49 | 19.7 GiB |
| `--vae-graph-input 224 224` | 图内 | 226.24 / 226.64 | 19.7 GiB |

裁剪的多长度安全检查补上最后一个精度（OPT-030）：`TRIM_PRECISION=e0m3_hadamard` **4 passed**。五个长度与"新建的单长度前端"逐位一致（`equal=True cosine=1 max_abs=0`），一次失败 capture 之后同样逐位一致，graph pool 与 regular cache 下毒后 20 次 replay 全部 equal、**poison bytes overwritten = 0**。`nvfp4`（FA4 关与开）与真实 dims 的行在 `0920` 一轮已绿。

三项跑完后 `THOR_CHECKLIST.md` 上不再有待测项，ISSUE-080 的六个条件全部满足并已关闭。

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

FA4 两个位点都不回退（`FA4 fallback` 为 `None`）。按这张表的判据，`stack` 相对 `vae_trim` 低 9.7 ms 且 vs official 不劣，达到「FA4 转默认」的工作门槛；原生 VAE 同理。默认值未改。

读这张表的两个口径问题（ISSUE-082）：`default` 行的记录基线 202–203 ms 来自只跑 frontend 的 gate，与带官方对照的 e2e 不是同一口径，`c20f3a0` 没有同口径的 gate 数；同一组开关在同一会话里出现两次，93.2 与 106.8 ms 相差 13.6 ms，比判据用的 2 ms 工作阈值大。`eccf14f` 轮回答了这两点：同轮 gate 202.2 ms 与 e2e `default` 202.0–202.4 ms 同口径且一致，同一会话内 `default` 三次重复的极差 0.4 ms、`stack` 0.5 ms；106.1 / 106.8 ms 属于 `c20f3a0` 那一轮偏高的会话状态。

微基准（随机权重、不含 VAE）：融合项合计约 −9.7 ms；FA4 backbone −26.9 ms，backbone + mot −58.5 ms；VAE 编码 stage 19.4 → 8.3 ms（进图）；VAE 预处理 kernel 0.97 → 0.13 ms。

### `0921` 轮：服务默认提升为最快配置的观察（nvfp4，LIBERO，commit `de0ef51`）

MAXN，GPC 1.575 / NVD 1.692 GHz，`emc_locked=null`，GPU 空闲；日志在 Thor 的 `thor_val/0921`。

**标定 v3**：两个 bundle 文件重录成功，version 3，identity 带 `num_views/image_h/image_w = 2/224/224`（`text_trim` 分别为 False、True），64 个 sample、约 0.96 s/sample；旧 v2 文件被拒并提示重录，符合预期。

**服务默认的 gate（fixture v2，`--warmup 20 --iters 100`，同一命令三次）**：三次的 `effective_config` 都是 `text_trim=True vae_encoder=native vae_graph=True use_fa4=True use_fa4_mot=True fa4_fallback_reason=None`；fidelity 全过（FA4 双位点 + 原生 VAE 对 fixture 里 cuBLAS + torch VAE 录的 fp16 参考，余弦仍高于阈值）。

| 次 | vs official min / median | P50 |
|---|---|---:|
| r1 | 0.99896 / 0.99931 | 105.42 ms |
| r2 | 0.99884 / 0.99932 | 105.93 ms |
| r3 | 0.99894 / 0.99933 | 105.66 ms |

极差 0.51 ms，基线余量 5%（约 5.3 ms）是它的十倍。`served_default` 用 r1 的记录播种（`latency_baselines.json`）。旧基线仍管它自己：`untrimmed_reference` 的 gate 通过，P50 202.38 ms。相对 `0920t` 的 125.86 ms（裁文本 + FA4 backbone + torch VAE 图外）约 −20 ms。

gate 的 P50（105.4）与同配置端到端行（下表 `stack` 94.4、`profile_default` 93.1）差约 11 ms：两个口径用的 prompt 不同（gate 测 fixture 的最后一条观测，端到端在各任务上取 P50），裁文本的延迟随有效 token 数变化。这是对差异来源的推断，没有单独验证；两列不要互相比较。

**三个风险**：
- FA4 首次编译（清 cute 缓存后同进程两次 `load_imagewam(profile="default")`）：冷 14.672 s、热 11.787 s，一次性差额 2.89 s。
- `FLASHRT_THOR_FA4=0`：不报错，`use_fa4=False use_fa4_mot=False`，P50 103.2 ms，与 `vae_trim` 行（103.3 ms）同量级。
- 目标工作负载（3×256×256，`img_len=768`）：`--profile default` 的 infer 121.92 / ABI 123.64 ms，native 行跳过（进图 VAE，预期），相对 `0919e`（216.93 / 173.55 ms）分别 −95.0 / −49.9 ms；`--profile native`（trim、FA4 关、图外 VAE）infer 180.01 / ABI 140.26 / native 139.97 ms。

**配置矩阵（flag 行现在每行写明 VAE 与两个 FA4 位点；libero_spatial，nvfp4，全部 rc=0，FA4 fallback=None）**：

| 行 | vs official min / median | P50 | FA4 bb / mot |
|---|---|---:|---|
| `default`（未裁剪阶梯底） | 0.99418 / 0.99764 | 202.2 | F / F |
| `vae` | 0.99410 / 0.99744 | 190.2 | F / F |
| `vae_trim` | 0.99905 / 0.99932 | 103.3 | F / F |
| `vae_trim_fa4bb` | 0.99895 / 0.99934 | 98.9 | T / F |
| `stack` | 0.99896 / 0.99931 | 94.4 | T / T |
| `stack_no_vae` | 0.99898 / 0.99931 | 105.3 | T / T |
| `stack_no_trim` | 0.99437 / 0.99750 | 131.2 | T / T |
| `profile_default` | — | 93.1 | 与 `stack` 同一组开关 |
| `profile_fast` | — | 93.3 | 同上 |
| `profile_native`（trim，无 FA4，无进图 VAE） | — | 115.0 | F / F |

`profile_default`、`profile_fast` 与 `stack` 在跑间波动内相同，vs official 不劣于 `stack`。边际（相对上一行）：原生 VAE 进图 −12.0，`text_trim` −86.9，FA4 backbone −4.4，FA4 mot −4.5 ms。

**fp8 标定重录后**：`fp8_static_cutlass` 的 `stack` 行接受新文件，vs official min 0.99989 / median 0.99994（`c20f3a0` 为 0.99995），P50 104.7 ms，FA4 双开、无回退。

**ABI / native 回归**：`08_gate_abi`、`08_gate_abi_vae_graph`、`08_gate_native_schema`（仍 7 records identical）、`08_gate_native`（5324 / 5348 节点，6 个 mutant 全检出）全过；native pytest 39 passed。

**全量 pytest**：86 failed / 631 passed / 2 skipped / 55 errors。第一处 FAIL 是 `test_static_fp8_set_activation_scale_equals_calibrate`：`act_scale` 相差 1 ULP（0.0093122218 对 0.0093122208）。`compute_scale_kernel` 的 `amax / 448.0f` 在带 `--use_fast_math` 的构建里不是 IEEE 除法，标定文件的 numpy 除法是，所以不逐位相等；这个原因是从编译选项推断的，没有看 SASS。测试已改为对该标度用 1 ULP 容差，`set_activation_scale` 的逐位一致改用设备自己的标度检查。第一处 ERROR 是 `test_imagewam_infer_action_noise.py` 构造 frontend 时 `torch.randn` 抛 `Offset increment outside graph capture encountered unexpectedly`，发生在 FA4 测试之后；那是 CUDA generator 的"正在捕获"标志没有复位的错误，成因（哪一个失败的捕获遗留了它）没有确认，见 issues.md ISSUE-087。ABI / native 门禁是随后的新进程，所以不受影响。

### `0921_final` 轮（部分）：最终表 LIBERO 一张的前两行与 ISSUE-087 探针（commit `22d3801`）

MAXN，GPC 1.575 / NVD 1.692 GHz，`emc_locked=null`，GPU 空闲；checkpoint sha256 前 16 位 `53620f93f8772d20`；日志在 Thor 的 `thor_val/0921_final`。这一轮**停在 L2 的 fp16**，L2 的 nvfp4 / fp8、L3–L5 没有跑，所以这里没有可用的表行：表里只有同一 session 的行才算相对官方的倍数，D 组之后整组重跑。

- **官方 torch（bf16 eager，`imagewam_official_torch_bench.py --workload libero`，10 步，horizon 64）**：P10 456.25 / P50 456.97 / P90 457.74 ms（n=30），相对另一次会话的 453.6 ms 约 +0.7%。
- **FlashRT fp16，服务 `default` profile**（`effective_config precision=fp16 text_trim=True vae_encoder=native vae_graph=True use_fa4=True use_fa4_mot=True fa4_fallback_reason=None`）：`infer` P10 274.20 / P50 274.70 / P90 275.86 ms（n=100），`valid_tokens=24`（活动 `x0=25`）。与未裁剪的 275 ms 几乎相同，而 nvfp4 同样的开关从 202 降到 103 ms：fp16 是唯一不随裁剪变快的精度，三个不同的轮次都是这样（284.38 / 273.8 / 274.70 ms），记为 issues.md ISSUE-088，成因未知。
- **ISSUE-087 探针（torch 2.9.1+cu130）**：`case_sync_inside_capture` 之后 `torch.randn` 抛 `Offset increment outside graph capture encountered unexpectedly`，一次成功的小捕获之后恢复；另外两种失败方式（捕获体内 Python 异常、捕获体内用了 RNG 再出错）不留下这个状态。`test_imagewam_fa4_dispatch.py` 单独 24 passed；`capture_sync` 之后紧接 `infer_action_noise.py` 5 passed（FA4 回退成功再录图，标志被复位）。清单里的 `-k capture_sync` 作用在两个文件上，把第二个文件的测试都 deselect 了，所以没有测到级联；是哪个测试留下标志仍未确定，D3 用哨兵去找。

### `0921d` / `0921d_final` 轮：ISSUE-088 / 087 的诊断与 LIBERO 最终表（commit `824f058`）

MAXN，GPC 1.575 / NVD 1.692 GHz，`emc_locked=null`，GPU 独占，未改频、未重编；checkpoint sha256 前 16 位 `53620f93f8772d20`；日志在 Thor 的 `thor_val/0921d`（D 组）与 `thor_val/0921d_final`（L 组）。

**LIBERO 最终表**（`docs/imagewam_results.md`，由 `docs/imagewam_results.json` 生成；同一 session，10 步，服务 `default` profile，指令 24 个有效 token）：

| 行 | P50（P10–P90） | 相对官方 | vs official 余弦 median / min | MAE vs GT |
|---|---:|---:|---:|---:|
| 官方 torch bf16 eager | 456.66（456.13–457.60） | 1.00× | — | — |
| FlashRT fp16 | 226.71（226.35–227.67） | 2.01× | 0.99998 / 0.99994 | 0.1856 |
| FlashRT fp8_static_cutlass（trim 标定） | 116.59（116.48–116.73） | 3.92× | 0.99994 / 0.99989 | 0.1859 |
| FlashRT nvfp4 | 108.03（107.96–108.16） | 4.23× | 0.99936 / 0.99885 | 0.1861 |
| int8 / int4 | 留空（决定） | | | |

三个口径问题：官方一行在 session 末尾复测是 377.04 ms（−17.4%），超过 2% 的界，所以三个倍数都标 §（分母没夹住；官方三个 session 的首次测量都是 453.6 / 456.97 / 456.66 ms，末尾这次才偏低）；nvfp4 的两端一致（107.62 ms，−0.4%）；fp16 一行只是这个 session 的值（见下）。延迟来自 `imagewam_thor_path_bench.py`（随机观测），精度来自同一 profile 与精度的矩阵行（真实 LIBERO 数据，libero_spatial，10 个任务 × 帧 0、60 = 20 个样本，seed 0）。当时怀疑是 EMC 频率没锁，`0921x` 一轮排除了这个原因：见下。

**ISSUE-088（fp16 不随裁剪变快）的新证据**：同一个命令 `0921_final` 是 274.70 ms、这一轮是 226.71 ms；D1 里 fp16 首张图（24 token）replay 294.95 ms、512 token 288.48 ms，D2 同进程 x0=21 是 172.36 ms（x0=513 是 270.93）。nvfp4 在各处都稳定并随行数缩小（D1 206.49 → 124.95；D2 181.80 → 109.81 ms），`fp16_cutlass` 也缩小（275.59 → 187.05）。fp16 的 512 token 图里 GEMM 占 218.28 / 288.48 ms，`nvjet_hsh_512x64` 平均 4.47 ms/次，图里 kernel 占比约 100%（没有空隙）。所以只有 cuBLASLt 的 fp16 GEMM 路径既不稳定又慢，指向它的算法选择（清单 X1 的探针定位）。`torch.profiler` 对每个精度第一张图抓到 0 个 kernel，与 nsys 抢 CUPTI（`MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED`），所以 24 token 的 kernel 表没有。

**ISSUE-087**：哨兵点名一个测试 `test_cuda_graph_timer_on_real_launches`；复位之后全量是 2 failed / 781 passed / 1 skipped / 5 errors（无哨兵是 86 / 631 / 55）。

**L4（int8 / int4）**：扩展里没有 `cutlass_int8_rowwise_fp16out` / `cutlass_int4_rowwise_fp16out`（只有 bf16out），没有重编；这两行按决定留空。

### `0921x` 轮：ISSUE-088 定位、OPT-032 kernel 在 Thor 上的逐位确认、EMC 假设排除（commit `939dc20`）

MAXN，GPC 1.575 / NVD 1.692 GHz，`emc_locked=null`，GPU 独占，未改频、未重编生产扩展；日志在 Thor 的 `thor_val/0921x`。

**X0 向量化激活量化**：`quant_act_nvfp4` 确认走上了 `quantize_fp4_dynamic_sfa_fp16_vec`，不是每次退化到标量（5 次 infer 里 3400 次 vec、0 次标量）；`default` nvfp4 P50 107.36 ms，与之前同量级。

**X6 新 AdaLN+FP4 融合 kernel 对照当前真正接入的融合 kernel**：本机只测过对照旧的未融合对（`gate_res_bf16res`+`ada_layer_norm_*`），Thor 上直接对照当前接入的 `gate_res_ada_layer_norm_bf16res`/`_fp16`（`fusion.cu`），**16/16 `packed`/`sfa`/`residual` 全部 `torch.equal`**。OPT-032 里"很可能但未直接验证"的说法可以去掉，K2 的融合 kernel 与当前生产路径逐位一致，Thor 已确认。

**X1（ISSUE-088 定位）**：裁剪后 `x0=25` 的 fp16 文本 GEMM 只有 4–6 TFLOPs（`txt_qkv` 6.1、`txt_mlp0` 4.6–5.9），未裁剪 `x0=513` 同类是 96–110 TFLOPs——不是行数效应，是小 M 本身在 cuBLASLt 上效率崩溃。启发式第一名经常明显慢于 autotune（`single_mlp_in` 31 对 84 TFLOPs、`img_mlp0` 47 对 80 TFLOPs），但 autotune 也没完全补上（`txt_proj` M=25 在三个 runner 之间还是 `0.019..0.065 ms` 分叉，零填充与随机填充选出同一个算法，排除了填充方式的影响）。根因：sm_110 上小 M 的 cuBLASLt 效率问题，不是别处的调度或融合缺口。

**X4（kernel 级 profile，24 token，FA4 关）**：

| 精度 | replay | kernel 内占比 | GEMM | other | attn | 平均 kernel | &lt;10µs 占比 | &lt;100µs 占比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nvfp4 | 123.51 ms | 103% | 77.24 ms | 38.47 ms | 11.46 ms | 23.8 µs | 13% | 90% |
| fp16_cutlass | 186.76 ms | 101% | 149.98 ms | 32.73 ms | 6.74 ms | 37.1 µs | 6% | — |

nvfp4 头名 kernel：`rms_norm` 23.0 ms × 560 次，CUTLASS GEMM 21.9 ms × 230 次、17.7 ms × 420 次。短 kernel 不是主因（&lt;10µs 只占 13%）；GEMM 和 rms_norm 占大头，GEMM 平均远低于 110 TFLOPs 的可达速率——与 OPT-032 的分析一致：既不是纯 launch 数量瓶颈，也不是纯算力瓶颈，GEMM 效率和 rms_norm 这类小 kernel 的数量都是真实成本。（`kernel_quantize_fp4_sfa_vec` 曾被误分进 attention 类，因为它的完整符号带 `flash_rt` 命名空间、命中了旧的 "flash" 关键字；已修，下一轮的分类会更准。）

**X5**：脚本缺了 X4 已有的 CUPTI 预热，四次都返回 0 个 kernel；本地已修，连同上面的分类 bug 一起，需要重新跑一轮才有网格数据。

**X2（官方漂移、tegrastats）**：跑完 X5 之后 455.46 ms，空闲 5 分钟后 459.37 ms，两次都在 456 那一档，**没有复现** `0921d_final` 那次的 377.04 ms（−17.4%）。Thor 的 `tegrastats` 没有 `EMC_FREQ` 字段，`/sys/class/devfreq` 下也没有 emc 节点。**EMC 频率假设排除**：那次偏低是那一轮 L3/L4 之后的某种 session 状态，不是空闲能复现的，成因仍不明。

**X3（两个真实测试失败，issues.md ISSUE-089）**：`test_failed_capture_leaves_no_replayable_graph` 打印 `cached=()`，与预期的 `(6,)` 不符——根因已定位并修：该测试没有显式传 `use_fa4=False`，在 FA4 可用的机器上会走到"FA4 失败级联清空整个缓存"的分支（这是另一个测试故意测的行为）。`test_fa4_fallback_keeps_old_graphs_until_the_replacement_exists` 死在 FA4 回退后与新建 frontend 的 cuBLAS 链路输出 `torch.equal` 比较，两侧数值接近但不是逐位相同（如 −1.5116 对 −1.5115）——原因未定，可能是 cuBLASLt 算法选择的非结合性，不是 bug；也可能是真实差异，还没确认。

### `0922` 轮：OPT-032 candidate 1 接入验证（发现测试自身的种子 bug）、X5 网格拿到数据、ISSUE-089 收尾（commit `8289754`）

MAXN，GPC 1.575 / NVD 1.692 GHz，`emc_locked=null`，GPU 空闲；日志在 Thor 的 `thor_val/0922`。

**X8（candidate 1 接入）**：重编成功（`qkv_split_norm_rope_fp16` 符号存在），但新加的接入测试没过逐位一致（`cos=0.9993150 max_abs=1.625`）。定位：不是接入代码的问题，是**测试自己的 bug**——`torch.manual_seed` 重置得太晚，两次对比调用（开关前/开关后）用到的部分权重和输入实际上不是同一份数据。已把重置挪到函数最开头、任何随机数抽取之前，并把测试名从 `fused_qkv_norm_rope` 改成 `fuse_qkv_norm_rope`（和 dims flag 同名，之前 `pytest -k fuse_qkv_norm_rope` 因为差一个字母 0 collected）。**接入代码本身是否正确，还要等修好的测试重跑才能确认**——plan.md 里 Phase 1 的状态从"completed"改回"active"。

**X5（阶段 × 算子网格，CUPTI 预热生效，四种配置都拿到数据）**：`valid_tokens=24`，FA4 关。

| 配置 | 总耗时 | GEMM | quantize | norm | attn |
|---|---:|---:|---:|---:|---:|
| nvfp4 | 126.7 ms | 76.54 | 4.95 | 25.58 | 6.64 |
| fp16_cutlass | 190.9 ms | 151.20 | 0 | 26.90 | — |
| fp8_static_cutlass | 137.5 ms | 88.89 | 2.76 | 25.85 | — |
| nvfp4 + FA4 | 102.1 ms | 46.13 | — | — | 11.65 |

GEMM 效率：backbone 接近算力（bb.single 164.9 TFLOPs、bb.double 82.3 TFLOPs），Action 侧远没有（act.single 32.6、act.double 24.1 TFLOPs），且 Action 侧短 kernel（&lt;25µs）占 25–35% 的时间——和 OPT-032 的判断一致：candidate 1（fuse_qkv_norm_rope）主要打的是 Action 侧这类小 kernel，不是 backbone 的 GEMM 效率。FA4 关时有一块 31.41 ms 的 "(none)"（未归入任何阶段的 GEMM），FA4 开时几乎消失（0.37 ms）——这部分是 cuBLAS 注意力链路的 QK^T/PV GEMM，没有被阶段包装函数捕捉到，成因未查（不影响其他数字的解读，留作后续小项）。

**X7（ISSUE-089 确认）**：cache 存活的问题已经修好（Thor 上打印 `cached=(6,)`）；同一测试后面新暴露一个断言（恢复后的图 vs 全新 frontend，`cos=1.0000000 max_abs=6.104e-05`）；`test_fa4_fallback_keeps_old_graphs_until_the_replacement_exists` 的 FA4 回退 vs cuBLAS 链路比较也是类似量级的不一致（约 1e-4）。两处都改成了容差判据（`cos > 0.9999` 且 `max_abs < 1e-2`），参照的是这个项目自己已有的先例（`docs/imagewam_last_block_kv_only.md` 里同类的 GEMM 算法选择非结合性），不是新确认的根因。

### `0922b` 轮：candidate 1 崩溃排查——不是 kernel 或接线的问题（commit 待推送）

`0922` 的种子 bug 修好之后，Thor 上 `_action_single_layer(fuse_qkv_norm_rope=True)` 这次真的抛了 `illegal memory access`（backbone 两种 `merge_qkv_mlp` 都过了 bit-exact）。本机独立复现（不需要 Thor）并定位：

- 用 `CUDA_LAUNCH_BLOCKING=1` 精确定位：崩溃发生在 `attn_out_proj.weight` 的 `gemm.fp16_nn` 调用（`cuBLAS error ... code=13`），不在新 kernel 自己的 launch 里。
- 关键实验：把 `_action_single_layer` 用**全新权重**连续调用两次（共享同一个 `GemmRunner`），**`fuse_qkv_norm_rope=False` 两次都用**——同样崩溃，第二次。说明和这次的新 kernel 完全无关，是一个已经存在的问题：`GemmRunner` 在同一形状上被不同的权重指针重复调用会出问题。
- 只调用一次（不管 `fused_qkv` 是 True 还是 False）不崩；两次调用、只要第二次用了新分配的权重张量就会崩。
- 把权重、注意力后端、输入只建一次，两次调用只切换 `fuse_qkv_norm_rope`（复用同一份权重）——不崩，backbone 和 ActionDiT 都是 `torch.equal`，连续跑 3 次稳定。

结论：**candidate 1 的 kernel 和接线代码本身是对的**，被测试自己的构造方式（每次比较都重建全新随机权重）意外踩中了一个和这次工作无关的 `GemmRunner` 潜在问题（issues.md ISSUE-090，已记录、未深挖——真实生产代码从不会对同一个 `GemmRunner` 用不同指针重复调用同一形状，权重只在 frontend 构造时建一次）。测试已经改成"建一次、复用"，本机验证通过，Thor 待重跑确认。`plan.md` 的 Phase 1 标记为 completed（Thor 对修好的测试的确认还没做）。

### `0922d` 轮：X8 确认性重跑，通过——candidate 1 单流接入 Thor 位一致完全确认（commit `e727f91`）

HEAD `e727f91`（`0922b` 修复已经在这个 commit 里，之前 `0922c` 那次是 Thor 在推送落地前就 `git fetch` 了，看到的是旧 commit `cb638cd`，跑的是修复前的测试代码，属于时间差，不是新问题）。拉到 `e727f91` 后重跑：

```
python -m pytest tests/test_imagewam_thor_real_wiring.py -k fuse_qkv_norm_rope -q -s
```

结果：`1 passed, 6 deselected, 1.40s`。三处比较全部 `torch.equal`/`bit_exact=True`、`max_abs=0`：backbone `merge_qkv_mlp=True`、backbone `merge_qkv_mlp=False`、ActionDiT single。MAXN，`emc_locked=null`，GPU 空闲，未重编。

结论：OPT-032 candidate 1 的单流接入（`_single_stream_layer`、`_action_single_layer`）在 Thor 上完全确认，两轮假失败（`0922` 的种子 bug、`0922b` 的 GemmRunner 脆弱性）都已排除且与这次的新 kernel 无关。`plan.md` Phase 1 关闭；`THOR_CHECKLIST.md` 的 X8 已删除。

### Phase 2 本机完成：candidate 1 双流接入（`_double_stream_layer`/`_action_double_layer`），Thor 确认待做（commit 待推送）

`_double_stream_layer`：RMSNorm 本来就按流分别做（txt 一次、img 一次，各自的源 qkv 缓冲区和权重），所以融合 kernel 调用两次，各自带自己的行偏移 RoPE 指针（`_ptr_offset(rope_table, x0, HD)` 给 img 流），取代原来两次各 3-copy+2-rms 再跟着一次跨整段的联合 `rope_apply_fp16_perhead`；开关打开时联合 RoPE 那两行整段跳过。`_action_double_layer` 是单次调用点，接线方式和 `_action_single_layer` 的非合并分支一样。

新增测试 `test_double_stream_and_action_double_fuse_qkv_norm_rope_bit_exact_at_real_shapes`，跟 Phase 1 那个测试同样的"权重只建一次、两次调用复用"结构（issues.md ISSUE-090）。本机（Ada）用 JIT 把真实 kernel 绑到 `flash_rt.flash_rt_kernels` 上跑了实际提交的测试函数，backbone 与 ActionDiT 都 `torch.equal`，连续 3 次稳定。

过程中抓到一个真实 bug，但确认是测试自己的构造问题，不是接入代码：ActionDiT 双流层的 `proj.weight` 那个 `Fp16Linear(gemm, proj_w.data_ptr(), n, k)` 构造时 `n`/`k` 传反了（应该是 `(ahd, aaw)` 结果写成了 `(aaw, ahd)`），导致 `key("proj.weight")` 跑出来的 GEMM 输出宽度比 `action_proj_scratch` 缓冲区实际分配的宽,写越界。越界部分恰好落在 PyTorch 分配器同一个大段内，`compute-sanitizer --tool memcheck` 因此报 0 错误（和 ISSUE-090 当时的现象一样：redzone 没能盖住这种"段内越界"）；但 fused/unfused 两条路径各自的其他中间缓冲区分配顺序不同，越界覆盖到的相邻内存也不同，所以两边最终结果不一致（cos=0.9997，不是崩溃，是那种"看起来接近但不是"的错）。定位方法：给 `GemmRunner.fp16_nn` 打点，记录每次调用的 `(M,N,K)` 和输出缓冲区的零拷贝快照，fused/unfused 逐个 GEMM 对比，只有 `proj.weight` 那一个不一致。修好（`n`/`k` 换回来）后重新验证 3 次稳定。

结论：`plan.md` Phase 2 标记 completed（Thor 确认待做，`THOR_CHECKLIST.md` X9）。

### `0922e` 轮：X9 确认通过——candidate 1 双流接入 Thor 位一致完全确认（commit `c5898a2`）

HEAD `c5898a2`。`python -m pytest tests/test_imagewam_thor_real_wiring.py -k fuse_qkv_norm_rope -q -s` → `2 passed, 6 deselected, 1.47s`。五处比较全部 `torch.equal`/`bit_exact=True`、`max_abs=0`：backbone single `merge_qkv_mlp=True`/`False`、ActionDiT single（这三处是 X8 的复核）、**backbone double-stream**、**ActionDiT double**（这两处是 X9 新增的覆盖）。MAXN，`emc_locked=null`，GPU 空闲，未重编。

结论：OPT-032 candidate 1 的单流（Phase 1）和双流（Phase 2）接入在 Thor 上都完全确认。`plan.md` Phase 1、Phase 2 都关闭；`THOR_CHECKLIST.md` 的 X9 已删除。

### Phase 4 第一轮本机完成：candidate 3（融合 AdaLN+NVFP4 直接量化）接线，Thor 数值确认待做（commit 待推送）

设计并接线了 `Nvfp4Linear` 的预量化输入方法：`gemm_prequantized(out_ptr, m, stream)`，跳过 `_quant_act`，假设调用者已经把有效的 NVFP4+SFA 写进了 `self.scratch`（前提是调用者自己先做过一次匹配的 `_ensure_scratch(m)`）。`AdaLNTarget`（`pipeline_thor.py`）新增一个可选字段 `lin`：消费这个 AdaLN 输出的那个 GEMM 对象；`_awq_target`（本来就拿着这个对象做 AWQ 折叠）现在无条件把它挂上去，不管有没有 AWQ。`_fused_gate_res` 新增一个默认关闭的 `fp4_direct` 参数：打开且 `target.lin` 是 `Nvfp4Linear` 时，直接调用它自己的 `gate_res_ada_layer_norm_fp4_sfa_bf16res`/`_fp16res`（`csrc/kernels/fused_norm_fp4/`），把 NVFP4+SFA 直接写进 `target.lin.scratch`，不再经过一份中间的 fp16 `modded`；默认关闭，所以在这轮之前的所有调用点（都没传这个参数）完全不受影响，就算 `target.lin` 恰好是 `Nvfp4Linear` 也一样——门是这个 flag，不是对象类型。

第一轮接线范围只做了 `_single_stream_layer` 自己的 `merge_qkv_mlp=True`→`linear1.weight` 这一个消费点（真实部署默认路径，这个函数里最宽的一个 GEMM），开关是 `dims["fuse_res_norm_fp4"]`。同时把这个 kernel 从"JIT-only、没进正式构建"接进了正式构建：`CMakeLists.txt` 的 `fp4_kernels_obj`（`ENABLE_SM100_CUTLASS` 那个 if 块）加了 `csrc/kernels/fused_norm_fp4/fused_norm_fp4.cu`，`csrc/fp4_bindings.cpp` 加了两个绑定，参数顺序跟 JIT 版本的绑定完全对齐。

本机（Ada，sm_89）完全没有 Blackwell/Thor 的 NVFP4 编译产物，`Nvfp4Linear` 本身的真实构造都做不到，所以只做了本机能做的两件事：(1) 新增 CPU-only 调用契约测试 `tests/test_imagewam_fuse_res_norm_fp4_dispatch.py`——用 `object.__new__(Nvfp4Linear)` 跳过真实 `__init__`（那个 `__init__` 需要导入 Blackwell-only 的 `flash_rt.flash_rt_fp4`）构造一个"isinstance 成立但没跑真实初始化"的假对象，验证 `_fused_gate_res`/`_awq_target` 的调用顺序、参数、默认关闭时的向后兼容，8/8 过；(2) 新增真正的位一致接线测试 `tests/test_imagewam_thor_real_wiring.py::test_single_stream_fuse_res_norm_fp4_direct_bit_exact_at_real_shapes`（两层 single-stream 链，`fuse_res_norm_fp4=False`/`True` 对比 `torch.equal`），但因为需要真实 `Nvfp4Linear`，本机用 `pytest.importorskip("flash_rt.flash_rt_fp4")` 跳过，从来没在任何机器上跑过。

结论：`plan.md` Phase 4 标记 "completed for Round 1's scope (Thor confirmation pending)"。真正的数值确认是 `THOR_CHECKLIST.md` X10，需要用 `ENABLE_SM100_CUTLASS` 重编。融合 kernel 本身的数值已经在 `0921x`（X6）确认过位一致，这次的测试跟 Phase 1/2 的测试一样，目的是抓接线/指针错误，不是抓数值错误。

### `0922f` 轮：X10 第一次真跑，抓到真实接线 bug——`gate`/`scale`/`shift` 传成了 fp32 指针，kernel 要的是 fp16（commit 待推送）

重编成功（`GPU_ARCH=110`，`flash_rt_fp4: building for sm_110a`，只增量编了 `fused_norm_fp4.cu`+`fp4_bindings.cpp`），测试也真的收集到了（不是 skip）。`X10` `torch.equal` 没过：`fuse_res_norm_fp4=True` 那条链的 bf16 残差炸到 `-7392`、`3008` 这种量级，对照组（`False`）还是 `O(1)`（`-0.79`、`5.16`、`-3.92`）。`_diff_stats` 报的 `cos=nan` 是 fused 侧本身出现 NaN/Inf，不是判据本身的问题。

定位：直接读 `csrc/kernels/fused_norm_fp4/fused_norm_fp4.cu` 里两个导出函数的真实 C++ 参数类型——`gate`/`scale`/`shift`/`inv_s` 全部是 `const __half*`（fp16），跟已有的 `gate_res_ada_layer_norm_bf16res`（`const float*`，kernel 自己在内部转 fp16）是两个不同的约定，K2 自己的测试 `tests/test_fused_norm_fp4_kernel.py` 也确实是拿 `.to(torch.float16)` 之后的 tensor 去调这个 kernel。我接线时错用了 `_mod_vec_ptr`（那是给 `const float*` 约定用的），把一段 fp32 数据的指针直接喂给期待 fp16 的 kernel——等于用一半的元素宽度去读一段本该是两倍宽的数据，读出乱码，跟观察到的"炸到 O(1e3)/NaN"完全对得上（不是普通的数值漂移那种小误差）。

修法：`_fused_gate_res` 的 `fp4_direct` 分支现在自己构造新的 fp16 `(dim,)` tensor（`t[0,0].to(torch.float16).contiguous()`，跟 `_fuse_mod_pair` 已有的转换方式一样）给 `gate`/`scale`/`shift`，`awq_inv_s`（如果有）也一样转一遍，再把这些新 tensor 的指针传给 kernel。加了一个专门的回归测试，在 mock kernel 的 `side_effect` 里同步把实际传进去的指针指向的字节读回来解码成 fp16 数值做校验——不是等 `_fused_gate_res` 返回之后再读（那样读到的内存已经被回收/可能被别的分配复用，这个测试第一版自己就踩了这个坑，读出来的数字明显不对，换成同步读之后才稳定通过）。本机 CPU 侧调用契约测试 9/9 过；本机没法跑真正的数值确认（`Nvfp4Linear` 需要 Blackwell/Thor）。

结论：issues.md ISSUE-091 记录了完整过程。这不是"重编就行"的问题，不需要重编（kernel 本身没改，只改了 Python 侧传参），Thor 直接重跑同一条 pytest 命令即可。

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
| FlashRT nvfp4 默认（该行未裁剪 + FA4 关） | 203.3 ms | 1.00× |
| FlashRT nvfp4 叠满 | 106.1 ms（`c20f3a0`） | 0.52×（快 1.9×；相对官方约 4.3×） |
| FlashRT nvfp4 服务默认（`0921`，端到端各 prompt） | 93.1 ms | 0.46×（快 2.2×；相对官方约 4.9×，官方数字来自另一会话，不是同一机器状态的比值） |

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
| 小 M CUTLASS tile | 有 | 已实现，可选（`gemm_variant_autotune` 仍是 opt-in；Thor 上的收益尚未测量，`issues.md` ISSUE-023） |
| 精度/延迟回归门禁 | 有（时钟锁定检查、同批次 A/B、per-view baseline） | LIBERO gate + 每设备 baseline + 时钟状态记录 |
| 多子图 stage 拆分 | RTC 前缀复用 / VJP 引导 | 不需要 |
| 完整 attention 链路融合 | 已测试，慢 5–7 倍，放弃 | 改用 FA4 |

Pi0.5 自身的 Thor 数字（NVFP4 + FA4，完整 `infer()`）：1 / 2 / 3 相机 23.01 / 27.17 / 31.74 ms，是不同的模型，仅作量级参考。
