# Serafini 博士论文第 6 章：用 Green Contexts 限制 GPU 的 SM 数，可以降低合并访存对 CPU 的带宽干扰；Thor 上跨步写只用 1 个 SM 就能把 CPU 拖慢约 4.5–5 倍

- **Serafini 博士论文（第 6 章）** · 2026 · 博士论文（UNIMORE，第 XXXVIII 期）· [链接](https://iris.unimore.it/handle/11380/1403093)（PDF：`/retrieve/07522302-a25c-4b05-9214-dd7e0b4c108d/Serafini.pdf`，176 页）
- Demystifying Memory Interference: Analysis and Mitigation of the Memory Contention Problem in Heterogeneous Systems-on-Chip（意大利语题名：Smascherare l'Interferenza di Memoria …）
- 作者：Andrea Serafini；导师 Paolo Valente，共同导师 Andrea Marongiu，企业导师 Alessandro Biasci（Evidence S.R.L.）。IRIS 元数据：`DCTERMS.issued` = 2026-03-31，入库 2026-04-29；PDF 生成日期 2026-03-23。
- 同行评审版本：**第 6 章没有找到同行评审版本**。论文没有"发表列表"章节，也不自引；第 6 章结尾写 "Our current work is focusing on integrating time-domain and space-domain technique…"（§6.6）。另外两章有对应期刊论文（Crossref 核对）：
  - 第 4 章 ↔ L. Carletti, A. Serafini, G. Brilli, A. Capotondi, A. Biasci, P. Valente, A. Marongiu, "Taking a closer look at memory interference effects in commercial-off-the-shelf multicore SoCs", *Journal of Systems Architecture*, 2025-10, doi:10.1016/j.sysarc.2025.103487
  - 第 5 章 ↔ A. Serafini, A. Biasci, B. Morelli, P. Valente, A. Marongiu, "Synchronous VS asynchronous reconfiguration of Memory Bandwidth Management Schemes: A comparative analysis", *Journal of Systems Architecture*, 2025-10, doi:10.1016/j.sysarc.2025.103483
  - 检索过程：Crossref（作者 + "memory interference GPU"；"spatial GPU throttling green contexts"；"GPU-to-CPU memory interference green contexts"）；WebSearch 两次（"Serafini Valente Marongiu green contexts …"，"Serafini Green Contexts GPU-to-CPU …"）；DBLP 有反爬页面，未能查询（†：DBLP 未核对）。结果只命中论文 PDF 本身和无关工作（arXiv 2601.07600 Jetson 性能隔离、arXiv 2604.08451 等）。

## 论文做了什么

第 6 章提出一个"空间域"内存带宽管理方案（MBMS）：用 LD_PRELOAD 拦截 `cudaLaunchKernel`，按 CPU 关键任务声明的 QoS（0–100）选择一个 CUDA Green Context（SM 分区），把 GPU kernel 放到更少的 SM 上运行，以降低 GPU 的访存并行度和带宽，从而减轻 GPU 对 CPU 的内存干扰。在 Jetson AGX Orin 与 AGX Thor 上用合成流量生成器（读 / 写 / 读写 × stride 16 B / 512 B）和 PolyBench-ACC 评估；另写了一个用 `%smid` 丢弃线程的生成器，模拟 1 SM 粒度的"理想 GC"。主要发现：合并访存下限 SM 有效；Thor 上 512 B 跨步写在 1 个 SM 时 CPU slowdown 就接近最大值，限 SM 无效。

## 结构

- 第 1 章 引言（§1.2.3 概述第 6 章贡献，含"单 SM 跨步写导致最高 5× slowdown"）
- 第 2 章 背景（§2.3.5 Orin、§2.3.6 Thor 平台参数）
- 第 3 章 相关工作
- 第 4 章 CPU 核之间的多级内存干扰表征（TX2、ZU9EG）
- 第 5 章 MBMS 同步 / 异步重配置对比（ZCU102）
- 第 6 章 本笔记重点：§6.1 背景（GPU、CUDA、GC）；§6.2 设计与实现；§6.3 合成负载；§6.4 合成负载评估；§6.5 PolyBench-ACC 评估；§6.6 结论
- 第 7 章 结论、局限与未来工作

## 问题表述

- 场景：一个 CPU 关键任务（`cpu_synth`，内存密集）和一个在 GPU 上运行内存密集 kernel 的干扰任务；两者都是 CPU 进程，GPU 一侧只有一个干扰来源（§6.1.1）。没有截止期数值，以 slowdown 度量。
- 优化目标：保护 CPU 关键任务不受 GPU→CPU 内存干扰。QoS 模型："each CPU task specifies a QoS requirement in the range [0..100], where 0 allows unrestricted GPU execution and 100 requires GPU-induced interference to be minimal"；全局 QoS 取所有 CPU 任务的最大值（§6.1.1）。
- 指标：
  - CPU slowdown = 有干扰时 `cpu_synth` 每轮迭代平均时长 / 单独运行时平均时长；作者称标准差 < 2%（§6.4.4）。PolyBench 实验里改为"每次访存延迟"的比值（§6.5.1）。
  - GPU 带宽：用 NVIDIA Central Activity Monitor（actmon）读内存控制器的 active cycle 比例 × 标称带宽（§6.4.4）。**这不是 GPU 侧实测吞吐，而是"控制器忙比例"折算的值**。
  - GPU kernel slowdown：分区运行时间 / 单独运行时间（§6.5.2）。

## 核心观察

1. **合并访存时 GPU 几乎能打满带宽**。证据：§6.4.4 原文 "nearly saturates the physically available bandwidth on both platforms: 205GB/s and 273 GB/s"；Orin 读写 192 GB/s（93.6%），Thor 读 257 GB/s（94.1%）（Fig.6.9c、Fig.6.10a）。actmon 折算。
2. **合并写对 CPU 的干扰最大**：Orin 1.55×，Thor 2.5×（Fig.6.9b、Fig.6.10b，§6.4.4）。
3. **限 SM 能部分降低合并访存的干扰**：合并写 GC 最紧分区（2 SM）时 Orin 1.55×→1.2×，Thor 2.5×→1.6×；1 SM 模拟时 Orin 1.15×、Thor 1.2×（§6.4.4）。
4. **Thor 上跨步访问的干扰更严重，且带宽很低**：512 B 跨步读最大带宽 Thor 65 GB/s、Orin 120 GB/s；Thor 上 CPU slowdown 读 3×、写 5×、读写 3.1×（§6.4.4，Fig.6.12）。限 SM 后读降到 1.4×，读写降到 2.5×，写"insensitive to regulation"；原因："even a single SM, when performing trf=write strided memory accesses, saturates the main memory with memory requests"（§6.4.4）。Orin 上跨步读几乎无干扰（<1.05×），跨步写 1.45×→1.2×（GC），1 SM 时 1.1×。
5. **真实 kernel（PolyBench-ACC）**：Thor 上不调节时 CPU slowdown 最高约 2.2×（bicg_kernel2 2.16、gemver_kernel3 2.21、gesummv_kernel 2.2，Fig.6.15），2 SM 时最高约 1.5×（§6.5.2）；Orin 不调节最高约 1.4×，2 SM 时都低于 1.1×（Fig.6.13）。GPU kernel 自身慢到约 8×（Orin，16→2 SM）、约 10×（Thor，20→2 SM），与 SM 缩减比例一致（Fig.6.14、6.16）。MBMS 每次 launch 引入约 7.5 µs 开销（15 µs / 19 µs 的 kernel 慢 2× / 1.7×，§6.5.2）。
6. **adi_kernel5（合并写，kernel 偏小）在 Thor 上不受调节控制**：所有分区下 CPU slowdown 约 1.4×（Fig.6.15；正文写不调节时 1.5×）。作者解释为 kernel 占不满 GPU（§6.5.2）。

## 具体问题的回答

### 问题 1：Green Context 的 SM 粒度

- 原文（§6.2.2 "GPU SM Partitioning"，Fig.6.8 旁）："The smallest partition (i.e., the one that enforces maximum regulation) is composed by two SMs – a limitation imposed by CUDA GCs [58]. Partitions then grow two SMs at a time (4,6,8 SMs...) … Intuitively, for a GPU, the number of partitions is half the number of the SMs." 以及 "on a 20 SM GPU (with 10 partitions) … [90..100] are mapped to the smallest partition (2 SMs)"。§6.3.2 再次写："the smallest SM partition that can be implemented using CUDA GCs is made up of two SMs"。
- 适用设备：正文对 Orin 与 Thor 都按 2 SM 起、步长 2 处理；PolyBench 热力图列数 Orin 8 列（16…2）、Thor 10 列（20…2）（Fig.6.13–6.16）。
- 依据：仅引用 [58] "Cuda green contexts, 2025. Accessed: 2025-12-15"（NVIDIA 文档），**没有写 CUDA toolkit 或驱动版本**，也没有写用的是 `cuDevSmResourceSplitByCount` 还是其他 API、是否查询过 `minSmPartitionSize` / `smCoscheduledAlignment`。软件环境只给 Jetson Linux 版本（Thor 38.2，Linux 6.8.12）。
- 是否测过实际使用的 SM：**没有**。论文没有在 GC 路径上读取 `%smid` 或查询分区实际 SM 数。`%smid` 只在模拟 1 SM 的 `spmd_traffic_generator_conf_sm` 里用来丢弃线程（Algorithm 10）。作者还写 Orin 上"CUDA does not support GPU kernel profiling when using CUDA Green Contexts"（§6.5.2），说明没有用 profiler 核对。
- 现行 CUDA Driver API 文档（v13.5，Green Contexts 页，WebFetch 核对）："Compute Architecture 7.X, 8.X, and all Tegra SoC: The smCount must be a multiple of 2. The alignment (and default value of coscheduledSmCount) is 2." / "Compute Architecture 9.0+: … The alignment is 8." 另外 `cuDevSmResourceSplitByCount` "may lead to fewer groups created … due to … alignment and granularity requirements"。即文档对 Tegra 写的是 2，与论文一致，与本场景实测的 8 SM 不一致。
- 图中的间接证据（仅图，读数为目测）：Thor 合并读（Fig.6.10a），GC "2 SM" 带宽约 55 GB/s，低于 1 SM 模拟的约 65 GB/s 与 2 SM 模拟的约 120 GB/s；GC "8 SM" 约 210 GB/s，接近 4 SM 模拟的约 200 GB/s。如果驱动把 2 SM 请求实际放大成 8 SM，GC "2 SM" 的带宽应接近 8 SM 模拟的约 235 GB/s，图中并非如此。合并写（Fig.6.10b）GC 2 SM ≈ 模拟 2 SM（约 100 GB/s，slowdown 约 1.6），两条曲线吻合。
- 判断：可能的解释，均未核实——
  - (a) 请求数与实际得到的 SM 数不同：本场景若用 `SplitByCount` 并按 `smCoscheduledAlignment` 或 9.0+ 规则对齐，可能被上取整到 8；论文若用 `cuDevSmResourceSplit`（显式 smCount，不自动取整）则可得 2。两边用的 API / 标志位不同会导致结论不同。
  - (b) CUDA / 驱动版本差异：Jetson Linux 38.2 对应的 CUDA 与本场景所用版本可能不同，Tegra 规则可能在版本间变化。
  - (c) 论文的 GC "N SM" 实际吞吐低于 N 个 SM 的模拟值（合并读上约为一半），可能来自 MBMS worker 线程的 launch 开销（每 kernel 约 7.5 µs）和每线程只做一次访存的 kernel 结构（Algorithm 9），而非 SM 数本身；这一点使"GC 2 SM 确实只用了 2 个 SM"也无法从带宽曲线反推确认。
  - 要分辨，需要在 Thor 上直接查询 `minSmPartitionSize` / `smCoscheduledAlignment`，并在 GC 内 kernel 里记录 `%smid` 集合。

### 问题 2：Thor 上 GPU 带宽随 SM 数的曲线（Fig.6.10、Fig.6.12；数值除正文给出的端点外均为仅图目测）

| 模式 | GC（2,4,…,20 SM） | 1 SM 模拟（1…20 SM） | 何时饱和 |
|---|---|---|---|
| 合并读（stride 16 B） | 2: ~55，4: ~110，6: ~160，8: ~210，10: ~245，≥12: ~257（正文给出最大 257 GB/s） | 1: ~65，2: ~120，4: ~200，6–8: ~235（峰值），之后缓降到 20 SM ~210 | GC 约 10–12 SM 到平台；模拟约 6–8 SM 达峰 |
| 合并写 | 2: ~100，4: ~185（峰值），≥6: ~165 平台 | 1: ~50，2: ~100，3–20: ~130–155 | 约 4 SM；CPU slowdown 在 4 SM 已到 2.5× 的最大值 |
| 合并读写 | 2: ~100，4: ~180，≥8: ~235 平台 | 1: ~75，2: ~135，4–6: ~210，之后缓降到 ~180 | 约 6–8 SM |
| 跨步读（512 B） | 全程 ~45–65 GB/s（正文最大 65 GB/s） | ~25–45 GB/s | 1–2 SM 已到平台 |
| 跨步写 | 全程 ~30 GB/s | ~30 GB/s | 1 SM 已到平台；CPU slowdown GC 全程约 4.4–4.5（图），1 SM 模拟约 4.3，20 SM 模拟约 3.0 |
| 跨步读写 | ~50 → ~35 GB/s | ~35 → ~30 GB/s | 1–2 SM |

- 对 CPU slowdown（仅图）：合并读 GC 2 SM ~1.05，16 SM 达最大 ~1.4；合并读写 GC 2 SM ~1.2，最大 ~1.8。
- 注意：跨步写的正文数字为 "reaches 5×"，图中 Max Slowdown 线约 4.5；读为 "3×"，图中约 2.9。
- 注意：带宽是 actmon 折算值（控制器 active 比例 × 273 GB/s），跨步模式下"控制器忙"不等于有效字节吞吐。

### 问题 3：是否测 GPU 对 GPU、CPU 对 GPU 的干扰

- **GPU 对 GPU：没有测**。GPU 上始终只有一个干扰任务（合成生成器或单个 PolyBench 程序，"one at a time"，§6.4.3、§6.5.1）。GC 在文中只用于"限带宽"，作者明确区分了它在文献中的"spatial multitasking"用途（§6.1.2）但没有做多 GPU 任务实验。
- **CPU 对 GPU：没有作为研究对象**。仅有间接数据：PolyBench 实验中 GPU kernel 与单线程 `cpu_synth`（1 个 CPU 核，64 B 步长读）并发；Thor 20 SM（不调节）列中较长 kernel 的 GPU slowdown 为 1.0（如 mm2、gemm、syr2k，Fig.6.16），作者把该列偏离 1.0 的值归因于 MBMS 开销而非 CPU 干扰（§6.5.2）。判断：这只说明 1 个 CPU 核的读流量不明显拖慢这些 kernel，不能推广到本场景 14 核满载（本场景实测 CPU 负载使 decode 慢约 2.2×）。

### 问题 4：Thor 平台细节

- CPU：14 × Arm Neoverse-V3AE，单 cluster，2.6 GHz；每核 64 KB I + 64 KB D L1、2 MB L2；共享 16 MB L3；64 B cache line（§2.3.6）。
- GPU：Blackwell，20 SM，每 SM 128 core，1.57 GHz；"Each couple of SMs share up to 256KB of L1 cache"；L2 大小 "no precise information … at the time of writing"（§2.3.6）。每 SM 最多 1536 个活跃线程（§6.3.2）。
- 内存：128 GB LPDDR5，256-bit，最高 4266 MHz（§2.3.6）；标称带宽按 273 GB/s 计（§6.4.4）。判断：论文写 LPDDR5，NVIDIA 官方资料为 LPDDR5X，未核对。
- 软件：Jetson Linux 38.2，Linux 6.8.12（§6.4.1）。**JetPack 版本、CUDA 版本、驱动版本均未写**。判断：Jetson Linux 38.2 对应 JetPack 7.0 系列，未在论文中核对。
- DVFS："we disable CPU, GPU, and main memory frequency scaling"，并关闭 GUI（§6.4.1）。**没有写固定在哪个频率、nvpmodel 功耗模式或是否运行 jetson_clocks**。

## challenge

- 时间域方法的限制：PREM 类方法要改 CPU / GPU 任务；利用 GPU 周期空闲的方法时间粒度粗（§6 引言）。
- 分区手段：MIG 分区选择少；MPS "does not control which SMs each task is assigned to"（§6 引言，引 Bakita & Anderson ECRTS 2025）。
- GC 最小分区 2 SM，可能掩盖 1 SM 时的效果（§6.3.2）。
- GC 下无法用 profiler 分析 kernel（Orin，§6.5.2）。
- MBMS worker 线程导致短 kernel 结果方差大（§6.5.2 末段）。

## 机制

- 拦截：动态库重定义 `cudaLaunchKernel` 与 `cudaDeviceSynchronize`，通过 LD_PRELOAD 注入；被拦截的 launch 入队并立即返回（§6.2.2，Fig.6.7）。
- 调度：worker 线程初始化时创建所有 GC 分区（2,4,…,N SM），每取出一个 kernel 读一次共享内存中的 CPU QoS，取最大值，映射到分区（每 10 个 QoS 等级对应一个分区，20 SM 时 10 个分区），在该 GC 关联的 stream 上调用原始 `cudaLaunchKernel`（Algorithm 7）。
- 依赖：CUDA Green Contexts（官方 API，不需要改驱动或 kernel 源码）；不依赖 MPS、MIG 或未公开接口。代价是每 kernel 约 7.5 µs，以及 launch 经过一个额外线程。
- 1 SM 模拟器（仅用于评估，不是机制的一部分）：launch 满 occupancy 的 grid（每 SM 12 × 128 线程），`%smid` ≥ num_sm 的线程直接退出，其余线程按"虚拟 blockIdx"循环完成全部访存（Algorithm 10）。

## 评估

- 基线：不调节（全部 SM）；对比 GC 各分区与 1 SM 模拟。
- 负载：CPU 侧 `cpu_synth`（64 MB buffer，64 B 步长 `ldr`，1000 次迭代，§6.3.1）；GPU 侧 `spmd_traffic_generator`（128 线程 / block，16 B `ld/st.volatile.global.v2.u64`，trf ∈ {read, write, readwrite}，stride ∈ {16, 512} B，§6.4.2）与 PolyBench-ACC（EXTRALARGE_DATASET，每组合重复 10 次，§6.5.1–6.5.2）。
- 主要结果：见"核心观察"。
- 消融：没有单独的消融；GC 与 1 SM 模拟的对比起到"粒度"消融的作用。

## 作者自述的局限

- 空间域调节对 Thor 上的跨步写无效，需要时间域与空间域结合，运行时识别病态访存模式并切换（§6.6；§7.2 第三条）。
- MBMS 的 worker 线程带来开销与方差，未来优化（§6.5.2 末段）。
- GC 下 kernel 加速现象（Orin 上 bicg_kernel2、gemver_kernel3 分区后变快）原因未分析，作者猜测与 L1 共享有关（§6.5.2）。

## 与本场景的前提对照

| 前提 | 论文 | 本场景（见 [survey.md](../survey.md) §0.2） | 是否成立 |
|---|---|---|---|
| 硬件 | Jetson AGX Thor，20 SM，128 GB，标称 273 GB/s | 同 | 成立 |
| 带宽上限 | GPU 合并读 actmon 折算 257 GB/s | GPU 拷贝实测 255 GB/s | 量级一致（测法不同） |
| 受保护方 | CPU 上的单线程内存密集任务 | GPU 上的执行器（ImageWAM） | 不成立 |
| 干扰方 | GPU 上单个合成 / PolyBench kernel 流 | GPU 上的 27B VLM（prefill + decode）+ CPU 内存负载 | 不成立 |
| GPU 对 GPU 争用 | 未测 | 核心现象 | 不成立 |
| 截止期 | 无，只有 slowdown | 200 / 533 ms 严格截止期 | 不成立 |
| SM 分区粒度 | GC 最小 2 SM、步长 2（引文档，未实测） | 实测 GC 8 SM 粒度、MPS 下限 8 SM | 不一致，原因待核实 |
| 软件版本 | Jetson Linux 38.2；CUDA 版本未写 | 见本场景环境 | 无法比对 |
| DVFS | 关闭调频（频率未写） | 未记录 | 未知 |

## 可以借用的 / 不能借用的

- 判断，可借用：
  - Thor 上"限 SM 不等于限带宽"的直接证据：合并写约 4 SM 就让 CPU slowdown 达到最大值；跨步写 1 SM 就接近最大值。这与本场景"按 SM 不重叠划分后，decode 仍造成干扰"一致，说明在 Thor 上 SM 数不是带宽的可靠控制量。
  - 带宽曲线的形状（合并读约 8–12 SM 饱和、写约 4 SM 饱和）可用于估计：给规划器 8 SM 时，其访存型 kernel（decode）基本已能接近打满带宽。
  - actmon 读内存控制器 active 比例，是在 Thor 上观测总带宽占用的一个低开销手段（与本场景的测法互补）。
  - 1 SM 模拟器（`%smid` 丢弃线程）可用来在 GC 粒度之外测"每 SM 带宽"。
- 判断，不能借用：
  - MBMS 本身：保护 CPU 任务的 QoS 模型、LD_PRELOAD + worker 线程（每 kernel 约 7.5 µs，执行器单次 37.5 ms 内 kernel 数量多时开销不可忽略）、以及"缩 SM 换带宽"的思路——对规划器缩 SM 会直接拉长 L（本场景实测：8 SM 时长上下文 L > 5.33 s），而论文的被调节方没有延迟要求。
  - 论文的 GC 2 SM 粒度结论不能直接用于本场景，因为未实测分区实际 SM 数、未给 CUDA 版本，且与本场景实测 8 SM 冲突。

## 待核实

- 论文使用的 CUDA / 驱动版本、GC API（`cuDevSmResourceSplitByCount` 还是 `cuDevSmResourceSplit`）与标志位。
- 在 Thor 上 `minSmPartitionSize` / `smCoscheduledAlignment` 的实际返回值，以及 GC 内 kernel 的 `%smid` 集合（用来解释 2 SM 与 8 SM 的差异）。
- 关闭 DVFS 后的具体频率与 nvpmodel 模式（论文未写）。
- Thor 内存类型 LPDDR5 vs LPDDR5X（论文写 LPDDR5）。
- DBLP 被反爬拦截，作者的完整发表列表未在 DBLP 上核对（†）。
- Fig.6.9–6.12 的中间点数值仅为目测；论文没有给数据表。
