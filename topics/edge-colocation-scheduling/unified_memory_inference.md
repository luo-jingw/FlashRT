# 统一内存设备上的 LLM / VLM / 多模型推理：框架与系统综述

检索日期：2026-11-18。场景：Jetson AGX Thor（128 GB LPDDR5X，273 GB/s，20 SM）上同时运行 27B LLM planner（prefill 数秒、decode 受带宽限制）和约 4B 的 action model（200 ms deadline）。硬件规格见 `research/topics/edge-colocation-scheduling/unified_memory_socs.md`。

约定：数字取自论文正文、官方文档或源码，括号内注明出处位置。`†` 表示未核实或来自二手来源。「判断」是本文的推断，不是文献结论。

## 主表

### A. 为统一内存设计的框架与运行时

| # | 工作 | 发表状态与链接 | 设备 | 利用了统一内存的哪一点 | 关键机制 | 关键数字（位置） | 多模型或 CPU+GPU 共享带宽 |
|---|---|---|---|---|---|---|---|
| A1 | MLX (Apple) | 开源框架；[文档 Unified Memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html) | Apple M 系列 | 零拷贝；CPU/GPU 并发 | 数组不绑定设备，每个 op 用 `stream=mx.cpu/mx.gpu` 选择执行单元；跨 stream 依赖自动插入 | 文档说明无依赖的 CPU/GPU op“可以（且很可能）并行”；没有性能数字 | 允许 CPU 和 GPU 并发，但不感知带宽（文档未提及 bandwidth） |
| A2 | llama.cpp（Metal / CUDA 后端） | 开源；源码 `ggml-metal-device.m`、`ggml-cuda.cu`（master @ `401563f1`） | Apple、Jetson、DGX Spark、AMD APU | Metal 后端零拷贝；CUDA 后端只做 UMA 容量统计 | Metal 后端在 `hasUnifiedMemory` 时用 `newBufferWithBytesNoCopy`（`MTLResourceStorageModeShared`）直接包装 mmap 权重；CUDA 后端在 `prop.integrated` 或 `GGML_CUDA_ENABLE_UNIFIED_MEMORY` 时从 `/proc/meminfo` 读取可用内存，该环境变量还会把分配改为 `cudaMallocManaged` | CUDA 源码中 `info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (#15034)`，因此 Jetson 上 iGPU 直接使用 host buffer 的零拷贝路径目前处于关闭状态 | 否 |
| A3 | TensorRT Edge-LLM (NVIDIA) | 开源 C++，JetPack 7.1 起提供；[overview](https://nvidia.github.io/TensorRT-Edge-LLM/overview.html)，[MLPerf blog 2026-09-16](https://developer.nvidia.com/blog/tensorrt-edge-llm-completes-the-mlperf-edge-agentic-benchmark-6-4x-faster-on-jetson-agx-thor/) | Jetson Thor、DRIVE Thor、DGX Spark、Jetson Orin | 主要利用容量：NVFP4 腾出内存给长上下文和 speculative 状态 | NVFP4 权重与激活、FP8 KV、paged attention、EAGLE3/MTP/DFlash speculative decoding、无 Python 依赖的 runtime | Qwen3.6-27B 跑在单台 AGX Thor（MAXN）上：52.33 tok/s，TTFT 中位数 247.12 ms，TPOT 中位数 14.68 ms；使用 tree MTP（8 步、16 节点验证树）（MLPerf v6.1 blog）。Blog 原文说低 batch decode “heavily bounded by DRAM bandwidth” | 否：overview 不涉及多模型并发、零拷贝或带宽 |
| A4 | CUDA for Tegra App Note (NVIDIA) | 官方文档 [docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html) | Xavier、Orin、Thor | 零拷贝内存类型选择与 coherence | 列出 device / pageable / pinned / registered / unified 五种分配在 CPU 和 iGPU 上的缓存属性（Table 1）；Xavier 起支持 I/O coherence（§3.1），Thor 起支持 full coherency（§3.2）；小 buffer 推荐 pinned（zero-copy）（§4.1） | 只给机制，没有数字 | 只说明 iGPU “shares the DRAM with CPU and other Tegra engines”（§3.3），没有带宽共享指导 |

### B. 利用统一内存或 SoC 异构处理器的研究系统

| # | 工作 | 发表状态与链接 | 设备 | 利用了统一内存的哪一点 | 关键机制 | 关键数字（位置） | 多模型或 CPU+GPU 共享带宽 |
|---|---|---|---|---|---|---|---|
| B1 | LLM in a flash (Apple) | ACL 2024；[arXiv 2312.11514](https://arxiv.org/abs/2312.11514) | Apple M1 Max 等 | 容量：模型超出 DRAM 时把 flash 当作下一级存储 | windowing 复用激活的神经元；row-column bundling 让 flash 读取连续成块 | DRAM 只放下一半模型时，推理速度比 naive 加载快 4–5×（CPU）和 20–25×（GPU）（Abstract） | 否 |
| B2 | PowerInfer-2 | arXiv 预印本 v3（2024-12）；[2406.06282](https://arxiv.org/abs/2406.06282) | OnePlus 12（Snapdragon 8 Gen 3，24 GB）、Ace 2 | CPU 与 NPU 共享带宽和地址空间 | 以 neuron cluster 为调度单位：稠密激活的 cluster 放 NPU，稀疏的放 CPU；flash I/O 与计算流水重叠 | 47B 模型 11.68 tok/s（Abstract）。实测带宽：CPU 单独 43.9 GB/s，NPU 单独 56 GB/s，两者同时运行合计 59.6 GB/s（§2，“XPU and Memory Bandwidth Sharing”） | 用 CPU+NPU 并发补满带宽；不涉及多模型 |
| B3 | llm.npu | ASPLOS 2025；[arXiv 2407.05858](https://arxiv.org/abs/2407.05858) | Xiaomi 14、Redmi K60 Pro | NPU 与 CPU 位于同一物理 DRAM，执行时免拷贝 | 把 prompt 切成定长 chunk 给 NPU 做 prefill；outlier 通道放到 CPU/GPU 上做 shadow 执行；子图乱序调度 | prefill 快 22.4×，能耗降低 30.7×（Abstract）。反例：CPU 和 NPU 的地址空间是分开的，shadow 执行需要在 CPU 侧再存一份权重，内存占用接近 2×；同步开销占端到端时延的 29.7%（§3.3） | 否 |
| B4 | HeteroInfer（arXiv v1 名为 HeteroLLM） | SOSP 2025；[arXiv 2501.14794](https://arxiv.org/abs/2501.14794) | Snapdragon 8 Gen 3 | 统一地址空间实现 GPU–NPU 快速同步；并发运行以提高总带宽 | prefill 和 decode 用不同的 tensor 切分方式；支持 layer 级与 tensor 级 GPU–NPU 并行；同步不经驱动，直接利用共享内存 | 单个处理器只能用到 40–45 GB/s，GPU+NPU 合计约 60 GB/s（理论峰值 68）（§1、§3）；decode 带宽从 GPU 单独时提高到 59.5 GB/s，达到可用最大值的 96%（§6）；比 GPU-only/NPU-only 引擎快 1.34–6.02×（Abstract）。与 GPU 密集的游戏同时运行时，prefill 慢 2.2%，decode 慢 17.7%（§1） | 单模型内 GPU+NPU 共享带宽；测量了对其他 app 的干扰，但没有调度它 |
| B5 | NANOMIND | ICLR 2026；[arXiv 2510.05109](https://arxiv.org/abs/2510.05109) | RK3566 自制板、RK3588、QCS6490 | 零拷贝跨加速器传递 embedding | 把 LMM 拆成 vision / projector / language / audio 几个 brick，分别映射到 GPU、NPU、DSP；Token-Aware Buffer Manager 负责缓冲 | 端到端能耗降低 42.3%（Abstract）；文中指出 NPU 和 GPU “lack DMA isolation and must coordinate access to shared memory”（§1） | 单个模型的多个模块并发；协调 buffer，但不调度带宽 |
| B6 | EdgeAgent | ASPLOS ’27（PDF 页眉）；[arXiv 2610.03394](https://arxiv.org/abs/2610.03394) | Apple M4（120 GB/s）、M4 Pro（273 GB/s） | 零拷贝 CPU+GPU tensor parallel；权重只保留一份 | 采用非对称内存布局的零拷贝 TP（CPU 使用 SME）；按序列可预测性分配 draft 长度，减少浪费的带宽；工具调用期间 suspend-and-yield | M4 上 GEMV 的 CPU+GPU 共同运行：4096×4096 比 GPU 单独快 1.36×，4096×128256 只有 0.99×，此时 GPU 已用到 120 GB/s 峰值的 80% 以上（Table 1，§2.4）；端到端 1.29× / 1.77×（Abstract） | 多 agent（同一模型、多请求）；把带宽墙当作约束，但不做带宽分配 |
| B7 | Agent.xpu | arXiv 预印本 v2（2026-01）；[2506.24045](https://arxiv.org/abs/2506.24045) | Intel Core Ultra 5 125H，DDR5 | 共享地址空间：抢占时无需交换 KV | prefill 放 NPU、decode 放 iGPU；kernel 级抢占；带宽感知 dispatch：当双方都是 memory-bound 且优先级不同时，推迟低优先级一方（§4.3） | 同时运行时，memory-bound 的 GEMV 最多慢 1.59×，compute-bound 的 GEMM 不超过 1.10×（Fig. 4）；reactive 时延降低至少 91%，proactive 吞吐提高 1.2–4.9×（v2 Abstract） | 是：跨 NPU/iGPU 按优先级做粗粒度带宽让步（多个 flow、同一模型族） |
| B8 | Dissecting UPM on MI300A | IISWC 2025；[arXiv 2508.12743](https://arxiv.org/abs/2508.12743) | AMD MI300A | 统一物理内存的分配器与 coherence 特性（HPC，非 LLM） | 测量各分配器的带宽、延迟、page fault 和 TLB 行为；给出移植策略 | GPU 带宽：`hipMalloc` 3.5–3.6 TB/s，`hipHostMalloc`/managed 为 2.1–2.2 TB/s，静态 `__managed__` 只有 103 GB/s；CPU 侧最高 208 GB/s（§4）；内存开销最多降低 44%（Abstract） | 否 |

### C. 统一内存设备上带宽争用的测量与管理

| # | 工作 | 发表状态与链接 | 设备 | 利用了统一内存的哪一点 | 关键机制 | 关键数字（位置） | 多模型或 CPU+GPU 共享带宽 |
|---|---|---|---|---|---|---|---|
| C1 | SERENO | OSDI 2026；[USENIX PDF](https://www.usenix.org/system/files/osdi26-xin.pdf) | OnePlus 13（Snapdragon 8 Elite）、8 Gen 3；源码分析扩展到 Dimensity 和 Exynos | 带宽调度：把后台 LLM 当作可以让出带宽的负载 | 用 draft 子图的执行时延作为内生的争用传感器；draft 阶段可被抢占，空缺的候选用 n-gram 补；verify 时按需在不同 batch 的预编译图之间切换（batch 越大，带宽越低）；插入 micro-sleep | 后台 LLM 使前台 jank 增加 153%，而 LLM 自身吞吐只下降 1.01%（prefill）和 1.64%（decode）（Abstract）；memory stall 增加 3.8×（§3）；根因是 NPU 流量在互连上享有硬件优先级，NPU 也不在 MPAM 的控制范围内（Table 1）；jank 最多降低 92.6%，同时吞吐最多提高 67.9%（Abstract）；verify 图的带宽（以 B=1 为 1.00）：B=8/16/32 时为 0.82/0.79/0.74（§5.1） | 是：调度 LLM（NPU）与前台 CPU/GPU 之间的带宽，全部在软件层完成 |
| C2 | HaX-CoNN | PPoPP 2024；[arXiv 2308.05869](https://arxiv.org/abs/2308.05869) | Jetson AGX Orin、Xavier、Snapdragon 865 | 共享内存争用建模 | 对并发 DNN 做逐层映射（GPU/DLA），映射时考虑层特性、共享内存争用和跨加速器切换开销 | 争用最多降低 45%，时延和吞吐分别最多改善 32% 和 29%（Abstract） | 是：多个 DNN 跨 GPU/DLA，用争用模型指导映射（CNN，非 LLM） |
| C3 | RT-Gang++ | arXiv v3（2024-04）；[2304.13110](https://arxiv.org/abs/2304.13110) | Jetson Nano（Tegra X1） | 内存控制器 QoS 硬件节流 | 实时 gang 调度；用内存控制器提供的 32 级 iGPU 带宽节流和 LLC 带宽节流保护 SLAM 任务 | 节流开到最高的第 31 级时，SLAM 接近完全隔离，DNN 时延约翻倍到约 73 ms；实际选用第 20 级以保住 DNN 的 20 Hz（§6.3） | 是：通过硬件节流 iGPU 来保护 CPU 实时任务。Orin/Thor 是否保留同样的节流接口未核实 † |
| C4 | Edge-Inference Governors Need Memory-Clock State（同一方向：DVFSLM [2609.13153](https://arxiv.org/abs/2609.13153)） | arXiv v3（2026-07）；[2606.16106](https://arxiv.org/abs/2606.16106) | Jetson Orin NX / AGX Orin | EMC（内存时钟）作为独立的 DVFS 域 | deadline 感知的 DVFS governor，状态中加入 EMC 时钟、decode horizon（KV 增长）和 GPU co-tenancy；用 contract-admission 做准入 | 不感知 EMC 的拟合在紧 deadline 下漏掉 25–28% 的周期，EMC-aware 时不超过 0.9%；有 GPU co-tenant 时 tenancy-blind 准入漏掉 32% 的 token（Abstract）。DVFSLM 联合调 GPU 与 EMC，每 token 能耗最多降低 12.4%（Abstract） | 部分：LLM decode 与 GPU co-tenant 共存时做准入控制，但不分配带宽 |

**已排除的工作**：
- MLC-LLM、ExecuTorch：本次检索没有看到针对统一内存或带宽的专门机制 †，未列入表格。
- Dovetail 和 DuoDecoding（draft 在 CPU、target 在 GPU）：两者假设离散显存。SoC 上的 speculative 路线由 SERENO 和 EdgeAgent 覆盖。
- UrgenGo：做 GPU kernel 的 deadline 调度，不涉及带宽。
- MI300A 上 LLM KV 管理：没有找到专门论文。GH200 已按硬件文件排除。

## 综合

### 统一内存让哪些离散 GPU 优化变得多余，哪些变得更重要

- **多余：显存与主存之间的搬运和容量拼接**（KV/权重 offload、H2D 预取、swap）。Agent.xpu 在抢占时无需交换 KV（§4.3）；MI300A 用统一模型后内存开销最多降低 44%（B8）。
- **仍然要做：分配器和缓存属性的选择**。同一块物理 DRAM，在 MI300A 上不同分配器的 GPU 带宽相差可达 35×（3.6 TB/s 对 103 GB/s，B8）。llama.cpp 因输出损坏，关闭了 CUDA iGPU 的零拷贝路径（A2）。
- **更重要：减少每个 token 读取的字节数**（低比特权重、speculative decoding）。TRT Edge-LLM 在 Thor 上跑 27B 达到 TPOT 14.68 ms（A3）。判断：按 NVFP4 约 4.5 bit/参数估算，一次完整读权重约 15 GB，需要约 56 ms，所以每次验证平均至少接受约 3.8 个 token，带宽墙只能靠 speculative decoding 摊薄。
- **更重要：把带宽视为全设备共享的资源**。单个处理器往往跑不满带宽，CPU+NPU 或 GPU+NPU 并发才能补满（B2：43.9/56 → 59.6 GB/s；B4：40–45 → 约 60 GB/s）。但当 GPU 单独已经接近峰值时，再加处理器就没有收益（B6：0.99×）。判断：Thor 的 GPU decode 很可能单独就逼近 273 GB/s，异构并发的增益空间小，争用代价大。
- **新出现的隐性调度者：互连和内存控制器的硬件优先级**。SERENO 发现 NPU 流量享有硬件特权，导致前台受损、LLM 几乎不受影响（C1）。因此在统一内存上，“谁先拿到带宽”由硬件 QoS 决定，而不由软件调度器决定。

### 文献是否把带宽当作调度对象；空白在哪里

- **已有的四种带宽调度形态**：
  - 软件让出：SERENO 在 draft 阶段抢占、切换 verify batch、插入 micro-sleep。
  - 优先级推迟：Agent.xpu 在双方都是 memory-bound 时推迟低优先级一方。
  - 争用感知映射：HaX-CoNN。
  - 硬件节流与时钟：RT-Gang++ 的 iGPU 节流、EMC-aware governor。
- **空白 1：争用对象**。现有工作处理的都是跨处理器争用（NPU 对 CPU/GPU，GPU 对 DLA），或同一模型的多个请求。没有工作处理两个 LLM 级模型共处同一块 GPU、一个是带宽型 decode、另一个有硬 deadline 的情况。
- **空白 2：平台**。没有工作针对 Thor/Blackwell iGPU。Thor 的 MIG profile 名为 `Ng.0gb`，只切分 SM，不切分内存（硬件文件 †）。RT-Gang++ 所用的内存控制器节流在 Thor 上是否开放也未知 †。
- **空白 3：调度粒度与目标**。现有的“让出”多以前台帧率或 jank 为目标，是反应式控制。没有工作把带宽当作可预算的量（每个 kernel 的字节数和时间）来保证另一个模型的 deadline。

### 判断：针对本场景的候选系统方向

- **Q1：planner decode 能否改造成带宽弹性任务？** 以 action model 剩余的 deadline slack 为信号，调节 planner 的 speculative 树大小、batch 和 micro-sleep，类似 SERENO 但在同一块 GPU 上。待验证：action model 时延被拉长到 solo 的多少倍以内时，planner 的 tok/s 损失最小。
- **Q2：只靠逐 kernel 的带宽需求模型，能否预测共存时的时延膨胀？** 依据是 Agent.xpu 的观察：compute-bound 的 GEMM 不超过 1.10×，memory-bound 的 GEMV 最多 1.59×。如果能预测，就可以把 planner 的 prefill（compute-bound）与 action model 的 memory-bound 段错开，同时避免 decode 与它重叠。待验证：在 Thor 上实测该模型的误差。
- **Q3：Thor 提供的硬件旋钮能否换来可预测的 deadline？** 候选旋钮包括 MIG 的 SM 切分、EMC 时钟、内存控制器 QoS 和节流。与纯软件让出相比，代价分别是什么？前提是先确认这些旋钮在 JetPack 7.x 中是否开放 †。
