# ImageWAM on FlashRT — Thor 部署状态总结

测试口径（除特别说明外）：Jetson AGX Thor，MAXN、GPU 锁频；LIBERO 配置——双相机 224×448（14×28 patch 网格）、8 维 proprio、10 步去噪、horizon=64、指令 16–31 个有效 token；真实 checkpoint；`infer()` 含 VAE 与 proprio，不含 Qwen3 文本编码。

## Pipeline 架构

一个 CUDA Graph 捕获的推理管线，用 C++/CUDA kernel + cuBLASLt/CUTLASS GEMM 通过裸指针调度，不走 PyTorch eager：

- **输入**：Qwen3-4B 文本编码（图外，每个 prompt 一次）；FLUX.2 VAE 图像编码（可选原生 NHWC 编码器，可进图）；proprio 状态（线性投影后插入文本 context，位置与官方一致）。
- **Backbone**：FLUX.2 DiT，5 层 double-stream + 20 层 single-stream，per-head QK-Norm、4 轴 RoPE、AdaLN modulation。
- **ActionDiT**：5 double + 20 single，10 步 flow-matching 去噪，官方 shift-based schedule。
- **输出**：反归一化到真实 action space。
- **执行**：prefill + 10 步 denoise 捕成一张 CUDA Graph，`infer()` 只是 replay；启用 `text_trim` 时每个有效文本长度一张图。
- **对外接口**：Python `infer()`；`frt_model_runtime_v1` ABI（`io="python"`）；原生 C++ overlay（`io="native"`，热路径不经过 Python）。

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
| `precision` | `nvfp4` | 精度/速度档位 | `fp8_static*` 需要校准文件 |
| `text_trim` | 关 | 按有效文本长度裁剪 | `runtime_surface()` / ABI 导出不支持（ABI 描述单一最大 shape 的图） |
| FA4（`FLASHRT_THOR_FA4`、`use_fa4_mot`） | 关 | 注意力 kernel | 首次调用编译，失败自动回退 |
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

三项叠加明显小于各项单独收益之和（−96 ms 对 −125 ms），`text_trim` 已经去掉了大部分 padding 上的注意力开销。

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
| **nvfp4** | 0.99933 (0.99887) | 0.186 | **106.1 ms** |
| fp8_static_cutlass（trim 校准） | 0.99995 (0.99990) | 0.186 | 115.3 ms |

官方参考 MAE 中位数 0.1855。

### FlashRT vs 官方 PyTorch 实现

官方为 bf16 eager（`nn.Linear`→cuBLASLt + SDPA），同输入配置，在另一次会话中测得。

| 路径 | P50 | 相对 FlashRT nvfp4 默认 |
|---|---:|---:|
| 官方 bf16 eager，端到端 | 453.6 ms | 2.2× |
| FlashRT nvfp4 默认 | 203.3 ms | 1.00× |
| FlashRT nvfp4 叠满 | 106.1 ms | 0.52×（快 1.9×；相对官方约 4.3×） |

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
