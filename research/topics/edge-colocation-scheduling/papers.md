# edge-colocation-scheduling 论文目录

- 快照日期：2026-11-17
- 范围：第一轮略读的 30 篇，按 [survey.md](survey.md) 的 A–D 四类分节；编号 A01–D30 是 survey.md 的引用编号。
- 年份：首次公开的年份（arXiv 或正式发表，取较早者）。
- 会议：取自论文首页的会议印记或 DOI、会议论文集页面、arXiv comment 或 Crossref；"预印本"表示没有找到发表处。标 † 的条目有未核实项，未核实项写在条目内。
- 数字取自正文并注明位置；只在图中出现的写"仅图"，只读到摘要的写"仅摘要"。"判断"是本综述的判断，不是论文的结论。
- "本场景"指 survey.md §0.2 的场景参照。
- 有精读笔记的 5 篇在条目末尾给出链接；条目与笔记不一致时以笔记为准。

## A. GPU 共享与抢占

### A01. REEF

**REEF** · 2022 · OSDI · [链接](https://www.usenix.org/conference/osdi22/presentation/han)
Microsecond-scale Preemption for Concurrent GPU-accelerated DNN Inferences
Mingcong Han et al.

- **场景**：AMD Radeon Instinct MI50（60 CU，16 GB，独立显卡）+ 8 核 i7；另将受限版 REEF-N 移植到 NVIDIA V100（§7.1）。负载为 DNN 推理：real-time（RT）任务按固定频率发请求（模拟自动驾驶摄像头），best-effort（BE）任务闭环持续发请求；自建基准 DISB 的 A–E 五种负载（Table 2）。模型为 VGG、ResNet、DenseNet、Inception、BERT 等小模型（单次推理毫秒级，摘要 Fig.1 中 RT 独占时延约 4 ms）。没有显式截止期，以"RT 时延接近独占"为目标。
- **目标与指标**：RT 端到端时延（相对 RT-Only）与总吞吐（RT+BE）；抢占延迟（preemption latency）；恢复时间。
- **核心观察**：
  - 多 stream 并发时 RT 尾时延从约 4 ms 升到接近 50 ms（§1，Fig.1b，MI50，VGG RT + ResNet BE）。
  - 等待式抢占（wait-based）的延迟随被抢占 kernel 数和 kernel 时长增长，268 µs（VGG）到 790 µs（§7.2，Fig.12b）。
  - DNN 推理 kernel 大多幂等（Table 1 统计），且单独运行时执行时间确定、可预测；VGG 的 kernel 时长 10–255 µs（Fig.10）。
- **challenge**：商用 GPU 不支持主动 kernel 抢占；host queue/device queue 中缓存数百个 kernel，清空它们的开销与队列长度成正比；并发执行时 BE 的干扰不可控。
- **机制**：(1) reset-based preemption：利用幂等性直接杀掉正在运行的 BE kernel、清空排队的 BE kernel，之后重跑，不保存上下文；配合异步内存回收与限制 device queue 容量（默认 4）。(2) dynamic kernel padding：把 BE kernel 与 RT kernel 编进同一个 launch，只用 RT 剩下的 CU，基于 kernel 时长可预测来挑选不拖慢 RT 的 BE kernel。(3) NVIDIA 闭源 GPU 上的 REEF-N 只能拦截 cuStreamCreate/cuKernelLaunch/cuStreamSynchronize，维护虚拟 host queue 并清空它，不能杀掉已在运行的 kernel（§4.4、§6）。
- **基线与评估**：RT-Only、SEQ（顺序/等待式）、GPUStreams。DISB A 下 SEQ/GPUStreams 吞吐 ×1.46/×1.66，但 RT 时延 ×1.95/×1.84；REEF RT 时延开销 0.5%，吞吐 ×1.60（§7.2，Fig.11）。DISB B 下 REEF RT 开销约 1%（约 60 µs）。reset-based 抢占延迟 35–38 µs（AMD，Fig.12b）。REEF-N 在 V100 上抢占延迟 71–288 µs，比 wait-based 快至多 12.3×（§7.5）。kernel padding 在 RT kernel 已占满 CU 时（IN3 85%、BERT 70% CU）收益仅约 6%（Fig.17）。
- **与本场景的关系**：
  - 成立：存在一类延迟关键（执行器）与一类可让步（规划器）任务；规划器的 forward 由大量 kernel 组成，"限制已提交但未执行的 kernel 数量"这一做法在 NVIDIA 上可用（REEF-N 只依赖拦截 launch）。
  - 不成立：RT 任务是 ms 级小 CNN，BE 也是小 CNN；本场景 BE 是 27B LLM，单次 prefill forward 约 1.17 s，其中 GEMM kernel 时长远大于 REEF 中的 kernel（判断：依据 Hummingbird §1 对 Llama-8B 在 A100 上 kernel 最长 7.49 ms 的测量，Thor 上 20 SM 会更长）。REEF 的主动杀 kernel 只在 AMD 上实现；NVIDIA 版仍要等运行中的 kernel 完成。论文不讨论内存带宽干扰；硬件为独立显卡。

### A02. Orion

**Orion** · 2024 · EuroSys · [链接](https://doi.org/10.1145/3627703.3629578)
Orion: Interference-aware, Fine-grained GPU Sharing for ML Applications
Foteini Strati et al.

- **场景**：NVIDIA V100-16GB（主），A100-40GB（泛化，§6）。一个高优先级（HP）作业 + 一个或多个 best-effort（BE）作业；HP 为推理（Apollo trace 或 Poisson 到达）或训练，BE 为离线推理或训练；模型为 ResNet、MobileNetV2、BERT、Transformer 等（Table 3）。无硬截止期，HP 以 p99/p95 时延相对独占为指标。
- **目标与指标**：HP 的 p99 时延保持接近独占（"ideal"），同时提高总吞吐/降低成本（cost savings = 2·Throughput_collocated / Throughput_dedicated）。
- **核心观察**：
  - DNN 作业由 10s–1000s µs 的算子组成，各自要么占满计算、要么占满带宽，平均利用率低：MobileNetV2 训练的计算吞吐利用率与内存带宽利用率均值分别低于 40% 和 55%（§1，Fig.1）。
  - 玩具实验（V100）：Conv2d（计算密集，89% 计算/20% 带宽）与 BN2d（访存密集，14% 计算/80% 带宽）同时运行比顺序快 1.41×；两个 Conv2d 并发无收益；两个 BN2d 并发只有小幅加速，因为都是访存密集（§3，Table 2）。
  - 现有技术（时间片、MPS、Streams、Tick-Tock）总吞吐远低于理想，REEF 保住 HP 但 BE 几乎不跑（Fig.2）。
- **challenge**：NVIDIA GPU 不能在提交后抢占 kernel、不能指定 SM 放置；BE kernel 一旦提交就会持续占用资源。
- **机制**：拦截 kernel launch，按 kernel 粒度调度。HP kernel 直接提交到最高优先级 stream；BE kernel 仅当（无 HP 在跑）或（BE kernel 所需 SM 数 < SM_THRESHOLD 且其 compute/memory 特征与当前 HP kernel 相反）时提交；另外跟踪已提交未完成的 BE kernel 预计总时长，超过 HP 请求时延的 DUR_THRESHOLD（默认 2.5%）就停止提交（§5.1，Listing 1）。离线用 Nsight Compute 给每个 kernel 打 compute/memory bound 标签。
- **基线与评估**：Temporal sharing、MPS、GPU Streams、Tick-Tock、REEF（策略重实现）、Ideal（独占）。推理+训练：Orion 让 HP p99 保持在理想值 14% 以内，比 REEF 低 2.33×；REEF 在 Apollo/Poisson 下分别为理想的 3.44×/2.5×（§6，Fig.6–7）。成本节省 1.26–1.49×（Table 4）。消融：只给 HP 最高 stream 优先级可降 p95 至多 25%，加上 compute/memory 配对再降 48%，再加 SM 数约束再降至多 54%（§6.4，Fig.14）。DUR_THRESHOLD 超过 3% 后 HP 性能近线性下降（§6.4）。作者在讨论中指出 LLM decode 访存密集，可与计算密集负载配对，但未评估（§7）。
- **与本场景的关系**：
  - 成立："stream 优先级本身只解决一部分，剩余干扰来自同类资源（带宽）竞争"与本场景"SM 不重叠后剩余干扰来自 decode 带宽"是同一现象；"限制在途 BE kernel 总时长"可在 NVIDIA 上纯软件实现。
  - 不成立：HP/BE 都是小模型、独立显卡、单进程拦截（需修改应用接入）；Orion 的 compute/memory 二分标签在 LLM decode 这种整体带宽密集负载下不够细（Hummingbird §3 实测 Orion 下 Llama-8B TPOT 波动比独占大 10.9×）；本场景执行器是周期性截止期任务，Orion 不处理截止期。

### A03. Clockwork

**Clockwork** · 2020 · OSDI · [链接](https://www.usenix.org/conference/osdi20/presentation/gujarati)
Serving DNNs like Clockwork: Performance Predictability from the Bottom Up
Arpan Gujarati et al.

- **场景**：多台 worker，每台 2×V100，服务数千个 DNN 模型（ResNet 等 TVM 编译的 CNN），Azure Functions 等生产 trace；请求级 SLO（10 ms–500 ms）。
- **目标与指标**：在给定 SLO 下的 goodput、SLO 满足率；与 batch 客户端共享时延迟敏感客户端的满足率（§6.4）。
- **核心观察**：
  - 单独运行时 DNN 推理时间确定：V100、batch 1，p99.99 与中位数相差 0.03% 以内（§2，Fig.2a）。
  - 用线程池并发跑推理吞吐最多 +25%，但尾时延上升约 100×（§2，Fig.2b）。
- **challenge**：并发、缓存、队列等"选择"在下层发生会引入不可预测性，并沿系统向上传播。
- **机制**：把所有选择集中到上层控制器（consolidate choice）：每个 GPU 一次只执行一个推理，控制器根据预测的执行时间与加载时间提前决定执行顺序，预计赶不上 SLO 的请求提前拒绝；batch 请求只在空闲或预期空闲时插入。
- **基线与评估**：Clipper、INFaaS。100 ms 目标下 99.9999% 请求达标（摘要）；SLO 收紧时 Clipper/INFaaS goodput 崩溃（§6.1，Fig.5）。与 batch 客户端共享时，延迟敏感客户端满足率不受影响，但 SLO <15 ms 时大量延迟敏感请求被提前拒绝（§6.4，Fig.8）。
- **与本场景的关系**：
  - 成立：执行器单独 37.5 ms、近似确定，"按预测时间提前安排"在本场景可用；"并发换吞吐、牺牲尾时延"与本场景不分 SM 时 p99 320 ms 的现象方向一致。
  - 不成立：Clockwork 完全放弃 GPU 内并发，靠多 GPU/多模型调度获得吞吐；本场景只有一个 GPU 且规划器 prefill 单块 forward 约 1.17 s，独占执行意味着执行器要等一整块（判断：会直接超过 200 ms 截止期，除非块被切小，而切小会改变输出）。拒绝请求在本场景不可接受。

### A04. Paella

**Paella** · 2023 · SOSP · [链接](https://doi.org/10.1145/3600006.3613163)
Paella: Low-latency Model Serving with Software-defined GPU Scheduling
Kelvin K. W. Ng et al.

- **场景**：Xeon Silver 4114 + NVIDIA Tesla T4（另测 P100，趋势相同，§7）；多客户端、多模型（ResNet、Inception、MobileNet 等，Table 2）；lognormal 到达（σ=2 或 1.5）。指标为 p99 时延与吞吐，无硬截止期。
- **目标与指标**：p99 job completion time、可持续负载（throughput before saturation）、服务框架开销。
- **核心观察**：
  - 一次性提交一个作业全部 kernel 会占满 GPU 的 32 个硬件队列，产生 HoL 阻塞；合成负载下最坏只用到 32/176 = 18% 的并发度，Paella 提升 goodput 2.2×（§2，Fig.2）。
  - Triton 服务框架开销对某些模型可与模型执行时间相当（Fig.3）。
  - CUDA stream 优先级与 MPS 仍受 HoL 阻塞与 FIFO 目标错配影响（§1）。
- **challenge**：GPU 硬件调度器是黑盒，host 不知道 kernel 何时开始/结束；通知与分发路径本身要足够轻。
- **机制**：TVM 编译期插桩，让每个 kernel 在 GPU 上开始/结束时通过共享内存通知 host；host 端单核 dispatcher 只在资源可用时逐个下发 kernel（"软件定义"调度），策略为 SRPT + deficit counter 公平阈值（§6）；客户端通信用共享内存轮询与协程。
- **基线与评估**：直接 CUDA（单/多 stream，多进程）、MPS、Triton v23.03、Clockwork（仅基线时延）。相对 Triton 可持续负载至多 58×、时延低 11×；仅调度算法本身比基线多 1.4× 负载、时延低 1.35×（§1、§7.1，Fig.11）。短/长作业混合下 ResNet-18 p99 比 CUDA-MS 低至多 3×（Fig.12）。
- **与本场景的关系**：
  - 成立："把还没提交的 kernel 留在 host 端由软件决定何时下发"在 NVIDIA 上可行，等价于控制在途 kernel 数。
  - 不成立：目标是多个短推理之间的平均/尾时延与公平，不区分硬截止期；所有模型用 TVM 编译插桩，对 TensorRT/CUTLASS/自写 kernel 的 LLM 不直接适用；小模型、独立显卡；不涉及带宽干扰。

### A05. XSched

**XSched** · 2025 · OSDI · [链接](https://www.usenix.org/conference/osdi25/presentation/shen-weihang)
XSched: Preemptive Scheduling for Diverse XPUs
Weihang Shen et al.

- **场景**：10 种 XPU：NVIDIA GV100/K40m/A100 等 GPU、AMD MI50、Intel GPU/NPU3720、Ascend 910b、NVIDIA DLA/PVA/OFA、Xilinx FPGA（Table 3）；含 NVIDIA Jetson Orin 与 Intel Core Ultra 两个 SoC 平台（§7.2）。负载为前台高优先级任务 + 后台任务；案例：多租户容器训练、视频会议（NPU 上背景替换 + whisper.cpp）、Triton/Paella 多模型推理（§8）。
- **目标与指标**：前台任务 P99 时延（相对单独运行）、抢占延迟、带宽划分精度、运行时开销与 CPU 占用。
- **核心观察**：
  - 硬件调度为非抢占 FCFS 或简单 RR；Intel NPU 上背景替换任务与语音转文字共跑时尾时延上升超过 20×（§1）。
  - 抢占延迟随所支持的层级变化：Lv1（只抢占未下发命令）的 P99 约为 8T（在途阈值 8，命令执行时间 T=0.5 ms）；Lv2（可撤回已下发未执行命令）约 1T；Lv3（中断正在执行的命令）在 GV100 上为 32 µs，与 T 无关（§7.3，Fig.11a）。
  - SoC 上 GPU 与 NPU 计算资源独立，但共享内存带宽与供电：只调度 NPU 任务时前台 P99 仍为单独运行的 1.67×/1.55×；把 GPU 后台访存任务一并纳入统一优先级后降到 1.18×/1.09×（§7.2，Fig.10）。
- **challenge**：不同 XPU 的抢占能力差异大且不断演进；软件栈闭源复杂；需要统一抽象支持与硬件无关的策略。
- **机制**：XQueue（可抢占命令队列）抽象 + 三级硬件模型。Lv1 用 progressive launching：只保持不超过阈值的在途命令，其余留在 host 缓冲（阈值 2–16，按设备调；阈值 >10 时开销 <1%，Fig.11c）。Lv2 在 NVIDIA GPU 上用 flushing（插桩让在途 kernel 检查标志提前返回）。Lv3 两种：TSG-based（把被抢占进程的 TSG timeslice 设为 0，仅进程间；源于 Tegra 上的 ioctl 做法）与 queue-based（通过一个未公开的 ioctl 触发 GPU 中断）（§6）。策略：固定优先级、带宽划分。
- **基线与评估**：各设备原生硬件调度、单独运行、TGS、Triton（含优先级设置）、Paella。固定优先级下前台 P99 降至多 2.10×；带宽划分开销平均 1.5%；SoC 跨 XPU 协同使前台 NPU P99 降至多 2.63×；容器场景比 TGS 多利用 2.74× GPU；Triton 集成 10 行代码，高优先级 BERT P99 从独占的 1.53× 降到 1.07×；与 Paella 性能相当（§1、§8，Fig.13–15）。
- **与本场景的关系**：
  - 成立：Jetson 类 SoC 上"计算隔离之后仍有共享内存带宽干扰"被直接测到，且通过"让高优先级到来时暂停低优先级任务的命令下发"缓解，这与本场景 SM 不重叠后 decode 带宽干扰同构；Lv1 progressive launching 只依赖拦截 API，可用于 CUDA；TSG timeslice 机制来自 Tegra 驱动。
  - 不成立：评估中的后台任务是小模型或合成访存任务，不是 27B LLM；Jetson Orin 实验中前台跑在 DLA 而非 GPU；Lv3 queue-based 依赖未公开 ioctl，在 Thor 驱动上是否可用论文未涉及；前台任务无硬截止期。
- **精读笔记**：[notes/2025-xsched.md](notes/2025-xsched.md)。卡片与笔记不一致处以笔记为准。

### A06. LithOS

**LithOS** · 2025 · SOSP · [链接](https://doi.org/10.1145/3731569.3764818)
LithOS: An Operating System for Efficient Machine Learning on GPUs
Patrick H. Coppock et al.

- **场景**：1×A100-40GB（108 SM）（§8 Testbed）；动机数据来自 Meta Ads 推理集群的 H100。HP 推理跑在 Triton（TensorRT、TensorRT-LLM、ONNX Runtime）：ResNet-50、RetinaNet、YOLOv4、Llama 3 8B、GPT-J 6B、BERT-Large；BE 为训练（ResNet-50、MobileNetV2、VGG-19、DLRM、BERT-Large）。SLO 取 MLPerf datacenter 时延约束，为基线端到端时延的 2.3–7.4×（§8，Table 2）。LLM 用 Azure trace。
- **目标与指标**：HP P99 时延与 SLO 达成率、goodput、总吞吐；right-sizing 的容量节省、DVFS 的能耗节省。
- **核心观察**：
  - 生产集群计算利用率平均 27%（§3.1，Fig.1；数值取自 arXiv v1 文本）。
  - 训练 kernel 常持续数毫秒，DLRM 超过 30 ms；大 prompt 的 LLM 推理 kernel 也达数毫秒（SOSP 版 §5，Fig.11）。
  - MPS 为吞吐牺牲 SLO；MIG 与线程数限制保住 SLO 但资源闲置（§2，Fig.3）。
  - 没有一个 TPC 数适合所有 kernel，需逐 kernel 选 TPC 数（Fig.12）。
- **challenge**：无硬件抢占下的细粒度调度；对应用透明（不改源码/框架/编译器）；长 kernel 的 HoL 阻塞；GPU 调频切换约 50 ms（arXiv v1 §4.6）。
- **机制**：拦截 CUDA Driver API（Rust，约 5000 行，基于 MPS）。(1) TPC Scheduler：基于 libsmctrl 的 TPC mask，按 TPC 给每个 kernel 分配计算单元，并允许借用空闲 TPC（TPC Stealing）（§7）。(2) Kernel Atomizer：不改源码/PTX，把一个 kernel 的 thread block 切成若干 atom 分批下发，缩短 HoL 阻塞并允许中途重新分配 TPC。(3) right-sizing：拟合 l = m/t + b 选最少 TPC。(4) 逐 kernel DVFS。
- **基线与评估**：时间片、MPS、stream 优先级、MIG、TGS（开源原版）、REEF 与 Orion（重实现）。推理叠加：尾时延比 MPS 低 13×，比最优 SotA 低 4×，goodput ×1.3（摘要，Fig.14–16）。推理+训练：MPS 下 HP 时延为理想的 5.83×，stream 优先级 2.89×，TGS 1.41×，REEF 尾时延达 8.93×；LithOS 在理想 20% 以内，比 MPS 平均低 4.7×，总吞吐比 TGS 高 1.35×（§8.1，Fig.17）。right-sizing 节省容量均值 26%（P99 +4%），DVFS 节能均值 26%（P99 +7%）（§8.2–8.3）。atomization 让 BE 吞吐损失约 10%（§8.4）。
- **与本场景的关系**：
  - 成立：HP 推理 + 长 kernel 的 BE 负载；BE 长 kernel 导致 HoL 阻塞，与本场景规划器 prefill 拖慢执行器同类；LithOS 的 TPC 分配与本场景 green context（8 SM 粒度）是同层机制，LithOS 粒度更细（TPC = 2 SM）且可逐 kernel 动态改变。
  - 不成立：只在 A100 上评估，GPU 有独立 HBM；论文不控制内存带宽（Hummingbird §1 也指出 TPC 级控制不解决 HBM/L2 干扰）；HP 的 SLO 是 MLPerf 的宽松倍数而非周期性截止期；BE 是训练而非另一个延迟也有要求的 LLM（本场景规划器也有 5.33 s 周期要求）。

### A07. GCAPS

**GCAPS** · 2024 · ECRTS · [链接](https://doi.org/10.4230/LIPIcs.ECRTS.2024.14)
GCAPS: GPU Context-Aware Preemptive Priority-Based Scheduling for Real-Time Tasks
Yidi Wang et al.

- **场景**：Jetson Xavier NX（Volta GPU）与 Jetson Orin Nano（Ampere GPU），Tegra 驱动（§7.2）。实时任务集：周期 30–500 ms、每任务 1–3 个 GPU 段的仿真任务集（Table 3）；实测案例为 7 个 CUDA Samples 任务（其中 2 个 best-effort，含一个 16 FPS 图形应用，Table 4）。有截止期与可调度性分析。
- **目标与指标**：可调度率、最坏响应时间（WCRT）上界、实测最大响应时间（MORT）、runlist 更新开销 ε 与 TSG 切换开销 θ。
- **核心观察**（本卡数字来自 arXiv v1 HTML，经 WebFetch 摘取）：
  - Tegra 默认驱动对各进程的 TSG（time-slice group）做轮转时间片，不看进程优先级，各 TSG 时间片看起来相同（§2）。
  - 实测 TSG 切换开销平均超过 200 µs（L = 1000 µs，Fig.13）；runlist 更新开销峰值约 1 ms（Fig.12）。
  - 基于同步协议（MPCP、FMLP+）的方法把 GPU 段当作不可抢占，导致长阻塞与优先级反转（§3，Fig.3）。
- **challenge**：GPU 时间片对上层不可见、不可控；切换开销要纳入分析；避免优先级反转与死锁。
- **机制**：在 GPU 段前后插 `gcapsGpuSegBegin/End` 宏，经 ioctl 通知驱动；修改约 300 行驱动，按任务优先级从 runlist 中移除低优先级 TSG 实现抢占；用 rt_mutex 减少阻塞；GPU 段可单独赋优先级（Audsley 方法）并给出响应时间分析（§5–6）。
- **基线与评估**：MPCP、FMLP+（busy/suspend 两种分析）、默认 TSG 轮转。可调度率提高至多 40%（摘要，Fig.8）；Xavier 案例中任务 1 的 MORT/WCRT 从 26.13/60 ms（tsg_rr_busy）降到 9.68/16 ms（gcaps_busy）（Table 5）；best-effort 任务 MORT 高于 FMLP+（§7.2）。
- **与本场景的关系**：
  - 成立：同为 Jetson（Tegra 驱动、TSG 时间片调度、统一内存 SoC）；有周期与截止期；进程级抢占与本场景两个独立进程（执行器、规划器）结构一致；"MPS 客户端优先级只有 2 档"之外，驱动级 runlist 控制是另一条进程级路径。
  - 不成立：任务是 CUDA Samples 小内核，GPU 段短；按段整体抢占在 TSG 切换上有约 200 µs 以上开销，对 37.5 ms 的执行器可接受，但论文未测 LLM 这种秒级 GPU 段；不分析内存带宽干扰（论文仅提到 zero-copy 不影响分析）；需要改驱动，Thor 的驱动版本是否兼容论文未涉及；时间切分意味着两者不同时在 GPU 上运行，不覆盖空间共享。

### A08. BLESS

**BLESS** · 2025 · EuroSys · [链接](https://doi.org/10.1145/3689031.3696070)
Improving GPU Sharing Performance through Adaptive Bubbleless Spatial-Temporal Sharing
Shulai Zhang et al.

- **场景**：NVIDIA A100（108 SM）；多个推理应用（VGG、ResNet、BERT、NasNet 等，TVM/PyTorch）各有 GPU 配额（quota，如 1/3 与 2/3 GPU），真实调用 trace；另有训练场景（§6，Table 1–2）。无截止期，配额即性能承诺。
- **目标与指标**：在保证每个应用"不低于按配额独立运行（ISO）"的前提下降低平均时延；时延偏差。
- **核心观察**：
  - 时间共享因 kernel 不可抢占而无法精确兑现配额；MPS 静态划分只分 SM，应用仍争内存带宽，且别人没用的 SM 也不能借（§1）。
  - Fig.1（A100，VGG11 配额 1/3 + ResNet50 配额 2/3）：某请求时延时间共享 17.1 ms、空间共享 11.5 ms，消除"气泡"后 10.1 ms。
  - 不同并发配置下 kernel 组的执行时间可预测（§4.4）。
- **challenge**：精确兑现配额；以低开销做运行时的细粒度资源重配（现有 stream / MPS context 原语粗且静态）。
- **机制**：每个客户端预先建立多个带 SM 亲和性的 MPS context（cuCtxCreate_v3 指定 SM 数）加一个不受限 context，运行时把 kernel 下发到不同 context 实现快速重配；把来自不同应用的 kernel 组成"kernel squad"，用离线 profiling 构建的估计器选出最优时空共享配置；squad 内前 c%（默认 50%）kernel 空间受限、其余不受限（§4.5）。认为代码变换类方法平均带来约 12% 额外时延，故不采用（§4.5.1）。
- **基线与评估**：Temporal、MIG、GSlice、Unbound（MPS/streams 不限）、Reef+（替换 padding 为 MPS）、Zico（训练）。平均时延比 SOTA 降 21.1–37.3%；两个 BERT 共置降 39.1%，四个降 41.2%（§1，§6）。
- **与本场景的关系**：
  - 成立：机制完全建立在 MPS 的 SM 受限 context 之上，并且"同一进程预建多个不同 SM 数的 context 并在 kernel 粒度切换"与本场景 MPS 百分比 / green context 的控制面一致；"MPS 只分 SM 不分带宽"的前提与本场景测得的现象相同。
  - 不成立：目标是公平配额而非一方有硬截止期；应用都是小推理模型；A100 独立显卡；按 kernel squad 同步切换 context 需要应用内拦截（判断：对 27B LLM 每层大量 kernel 的同步开销论文未测）。

### A09. Hummingbird

**Hummingbird** · 2026 · 预印本 · [链接](https://arxiv.org/abs/2601.04071)
Hummingbird: SLO-Oriented GPU Preemption at Microsecond-scale
Tiancheng Hu et al.

- **场景**：NVIDIA L40s、A100、H100，另有 2 台 DGX A100 多卡实验（§1 结果概述）。动机实验：高优先级为 llama.cpp 服务模式的 Llama-8B（8-bit，BurstGPT trace），低优先级为 Mistral-7B 离线摘要（batch 32）（§3）。评估覆盖 2 个高优先级任务（Llama-8B、Yi-34B serving）与 4 个低优先级任务（Mistral-7B、DeepseekMoE-16B 离线推理，ResNet-101、GPT-2 训练，Table 1）。SLO 为高优先级独占运行时的 P99 TTFT 与 P99 TPOT（§6.1）。
- **目标与指标**：高优先级 SLO 达成率；低优先级吞吐（TPS）；GPU 利用率。
- **核心观察**：
  - A100 上 Llama-8B 推理的 kernel 时长 5 µs（elementwise）到 7.49 ms（GEMM），导致等待式抢占的延迟不可预测（§1，Fig.3a）。
  - 跨 CNN/LLM/MLLM 的推理与训练，99.999% 的 thread block 执行时间在 390 µs 以内（§1，Fig.3b）。
  - Llama-8B / DeepSeek-16B 推理中超过 15% 的 GPU 时间是小气泡（访存操作与同步、通信、CPU 侧瓶颈）（§1，Fig.2a）。
  - 在 Orion 下高优先级 SLO 达成率比 REEF 低 5.4×、比独占低 8.1×；Orion 下 Llama-8B 的 TPOT 波动比 REEF 大 4.2×、比独占大 10.9×，原因是 decode 访存密集，Orion 以 60% 带宽利用率为界的 compute/memory 标签太粗，且被选为"计算密集"的 kernel 仍会占满带宽（§3，Fig.5）。
  - REEF 仍比独占差 1.5×（等待运行中的低优先级 kernel），且低优先级吞吐比 Orion 低 2.1×（§3，Fig.5a）。
  - Fig.1（概览，仅图）：MPS、Multi-streams、MIG、LithOS、Orion 的 SLO 达成率分别为 2.5%、7.2%、0.1%、31%、12.5%，REEF 72.3%，Hummingbird 98.4%。
- **challenge**：NVIDIA 不支持主动抢占；空间共享无法隔离 L2/HBM 带宽；低优先级被频繁抢占的同步与重启开销。
- **机制**：split-kernel：把低优先级 kernel 按 thread block 数切成多个小 kernel（每段为一波 block，平均 77 µs；400 µs 是 block 时长上界给出的抢占延迟上界，§4.2），形成 µs 级抢占点；高优先级到达时停止下发低优先级的后续分段；在高优先级执行期间的小气泡中插入低优先级分段（§4）。
- **基线与评估**：MPS、Multi-streams、MIG、Orion、LithOS（空间共享）；REEF（时间共享）；独占。SLO 达成率比空间共享 SotA 高 9.7×、比时间共享 SotA 高 3.5×；比独占下降不到 1%；低优先级吞吐比 REEF 高 2.4×（摘要）。正文单 GPU：SLO 达成率比 LithOS 高 5.6×、比 REEF 高 3.0×，低优先级吞吐为 REEF 的 1.9×（§6.1）；9.7× 是多 GPU 下相对 REEF（§6.3）。
- **与本场景的关系**：
  - 成立：两边都是 LLM 类负载；直接测量并归因了"空间共享下 LLM decode 带宽干扰"，与本场景按 SM 划分后剩余干扰的来源一致；按 block 切分低优先级 kernel 以控制占用时长是不改算法输出的做法（判断：切分只改变 block 下发批次，不改变每个 block 的计算，因此不同于本场景"切小 prefill 块会改变输出"的约束）。
  - 不成立：高优先级是 LLM 服务、低优先级是离线批处理，与本场景"执行器为小模型高优先级、规划器为 LLM 低优先级但也有周期要求"相反；独立 GPU、HBM；预印本，会议未定。
- **精读笔记**：[notes/2026-hummingbird.md](notes/2026-hummingbird.md)。卡片与笔记不一致处以笔记为准。

## B. LLM 服务的阶段调度

### B10. DistServe

**DistServe** · 2024 · OSDI '24 · [链接](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving
Yinmin Zhong et al.

- 场景：A100 集群（4 节点 × 8 × A100-80GB，§6.1）；OPT-13B/66B/175B；chatbot（ShareGPT）、code completion（HumanEval）、summarization（LongBench）。无硬截止期，只有统计 SLO。
- 目标与指标：最大化 per-GPU goodput = 满足 SLO attainment 目标（如 90 %）时每 GPU 可承受的最大请求率（§1）。SLO 是 TTFT 与 TPOT 双约束，按应用设定（Table 1：chatbot OPT-13B 0.25 s / 0.1 s；code completion OPT-66B 0.125 s / 0.2 s；summarization OPT-66B 15 s / 0.15 s）。用 SLO Scale 统一缩放两者。
- 核心观察：
  - 单 A100、13B、summary 类负载下，colocated 系统 90 % attainment 的 goodput 约 1.6 rps；prefill / decode 各占一卡时分别为 5.6 / 10 rps，2P1D 折合 3.3 rps/GPU，约 2.1×（§1, Figure 1）。
  - 往 decode batch 里加一个 prefill 会同时拖慢两者，prefill 越长 decode 越慢（Figure 2，仅图）。
  - chunked prefill 只能缓解不能消除干扰，且 N 块需加载 O(N²) 次 KV（§2.3）。
  - 把两阶段分开排队但仍同卡运行，优先任一阶段都会损害另一阶段延迟，"priority scheduling ineffective"（§2.3，定性论述）。
- challenge：分离后要为两阶段分别选 GPU 数与并行策略，并承受 KV 传输。
- 机制：prefill / decode 实例分离到不同 GPU；用模拟器 + 搜索为每阶段选并行配置与副本数（§4，Alg. 1/2）；按带宽放置以降低 KV 传输开销。
- 基线与评估：vLLM、DeepSpeed-MII。摘要与 §1：最多 7.4× 请求率或 12.6× 更紧 SLO，同时 >90 % 请求满足延迟约束。
- 与本场景的关系：
  - 成立：prefill 与 decode 计算特征不同、同卡共置相互干扰，这一前提与本场景一致；"优先级调度在同卡上无效"的论断针对同一模型两阶段，和本场景用 MPS 优先级保护另一个模型的结论不冲突（判断）。
  - 不成立：靠多 GPU 物理隔离，Thor 只有一块 20 SM 的 GPU；SLO 是统计 attainment，不是周期硬截止期；不涉及另一个模型。

### B11. Sarathi-Serve

**Sarathi-Serve** · 2024 · OSDI '24 · [链接](https://www.usenix.org/conference/osdi24/presentation/agrawal)
Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve
Amey Agrawal et al.

- 场景：Mistral-7B（1×A100）、Yi-34B（2×A100）、LLaMA2-70B（A40，PP）、Falcon-180B（2 PP × 4 TP，8×A100）（§1, §5）；openchat_sharegpt4、arxiv_summarization 两类负载。无硬截止期。
- 目标与指标：Capacity = 满足 P99 TBT SLO 且中位调度延迟不超过 2 s 时可承受的最大 QPS（§2.4, §5.1）。TTFT 取中位数，TBT 取 P99。SLO 定义为无 prefill 干扰时 decode 迭代时间（4k prefill、batch 32）的 5×（strict）和 25×（relaxed）（§5.1，Table 3：Mistral-7B 0.1 / 0.5 s；Yi-34B 0.2 / 1 s；LLaMA2-70B、Falcon-180B 1 / 5 s）。
- 核心观察：
  - prefill-prioritizing 调度（vLLM、Orca）产生 generation stall，可持续数秒（§1, Figure 1）。
  - decode batch 处于 memory-bound 区，可以附带一些 prefill token 而几乎不增加延迟（§3.1 Takeaway-2）。
  - prefill 吞吐在约 512 token 开始饱和（§4.1, Figure 4）。
  - 不切块的混合 batch 让 TBT 最多升 28.3×（§4.2, Figure 9）。
  - Falcon-180B 下 4k prompt 迭代约 1150 ms，batch 32 decode 迭代约 200 ms，PP 下形成约 950 ms 气泡（§3.3）。
  - chunked prefill 的额外开销（重复读 KV、kernel launch）见 Figure 14（仅图）。
- challenge：吞吐与 TBT 尾延迟的权衡；token budget 太小则 prefill 开销上升，太大则违反 TBT。
- 机制：chunked prefill + stall-free batching（Alg. 3）：每轮先放所有 decode，再放未完成的 prefill 块，最后在剩余 token budget 内加入新请求；token budget 由一次性 profiling 取"不违反 TBT SLO 的最大 token 数"，并考虑 tile quantization（§4.3）。
- 基线与评估：Orca、vLLM。strict SLO 下最多比 Orca 高 4.0×、比 vLLM 高 3.7× capacity（Yi-34B）；PP 下最多 6.3× / 4.3×（LLaMA2-70B）（§5.1）。摘要另给 Mistral-7B 2.6×、Falcon-180B 端到端 5.6×。
- 与本场景的关系：
  - 成立：用固定计算量的块来约束每次迭代时长，是在单设备上把 prefill 对他人的干扰做成有界量的直接手段；本场景已用 2048 token 块。
  - 不成立：本场景约束的是另一个模型（执行器）的周期延迟，不是同一 batch 中 decode 的 TBT；且已观察到块大小改变会改变输出，块长不能自由下调；token budget 由 TBT 推导，本场景对应的应是执行器截止期，Sarathi 未给出跨模型的推导方法。

### B12. semi-PD

**semi-PD** · 2025 · 预印本 · [链接](https://arxiv.org/abs/2504.19867)
semi-PD: Towards Efficient LLM Serving via Phase-Wise Disaggregated Computation and Unified Storage
Ke Hong et al.

- 场景：A100-80GB、A100-40GB、H200、A800 服务器（§7.1.1）；Llama3-8B/70B、Llama3.1-405B、DeepSeek-V2-Lite、DeepSeek-V3（Table 1）；ShareGPT、LongBench、MATH-500，Poisson 到达。无硬截止期。
- 目标与指标：TTFT / TPOT SLO，tight / loose 约为单请求延迟的 7.5× / 10×（Table 2，例：Llama3-8B ShareGPT tight 0.3 s / 0.15 s；LongBench tight 2.25 s / 0.13 s）；attainment 目标 >90 %；另报平均端到端延迟。
- 核心观察：
  - 分离部署中 prefill 实例最多只用 25 % KV 存储，最多 89.33 % 显存浪费（§3.1, Fig. 3）。
  - 存储耗尽使 P90 TPOT 升 3×（§3.1, Fig. 4）。
  - KV 传输开销约等于生成一个 token 的延迟（§3.2）。
  - DistServe 重规划需数分钟（含重载权重）（§3.3）。
  - 延迟模型：TTFT_x = a1/(x−λ) + b1，TPOT_y = a2/y + b2，x、y 为 SM 百分比，Fig. 10 给拟合（仅图）。
- challenge：在同一 GPU 内分开计算却共享存储，并能在运行时调整 SM 比例而不中断服务。
- 机制：prefill 与 decode 各为一个进程，用 NVIDIA MPS 百分比 (x, y) 划分 SM，允许 x + y > 100 %；常驻进程持有权重与 KV，通过 IPC 指针共享；延迟切换与异步切换（只重启已完成迭代的那个 worker）；原子化的 KV 分配避免 write-after-read 冲突（Fig. 7）；SLO 感知算法按窗口调整 (x, y)（§5, Alg. 1）。切换开销称"negligible"，未给测量值。
- 基线与评估：DistServe、vLLM v0.6.0（默认与 SplitFuse）、SGLang v0.4.4、Dynamo v0.1.1。DeepSeek 上平均端到端延迟降 1.27–2.58×（§7.2.1, Fig. 14）；Llama 上保持 >90 % attainment 时可承受请求率比 vLLM-S / vLLM-D / DistServe 高 1.55× / 1.72× / 1.62×（§7.2.2, Fig. 15）。
- 与本场景的关系：
  - 成立：机制形态最接近本场景——两进程 + MPS 百分比分 SM + 共享权重 / KV；"改 MPS 比例需要重启进程"的问题在本场景同样存在。
  - 不成立：不分析 prefill / decode 进程之间的带宽争用（本场景划分 SM 后剩余干扰正是 decode 带宽占用）；调度只用 FCFS，无优先级、无截止期；两进程同属一个模型，SLO 目标对称。

### B13. MuxWise（Drift）

**MuxWise（v1/v2 名为 Drift）** · 2026 · ASPLOS '26 † · [链接](https://arxiv.org/abs/2504.14489)
Towards High-Goodput LLM Serving with Prefill-decode Multiplexing（v3；v1/v2 标题为 Optimizing SLO-oriented LLM Serving with PD-Multiplexing）
Yukang Chen et al.（v3 作者顺序；v2 为 Weihao Cui 第一）

- 版本说明：本卡片的数字取自 v2；v3 有实质变化（平均 goodput 2.20×、最多 3.06×；争用 0–30 %；每组 green context 4 MB），以精读笔记为准。会议来自 v3 PDF 首页的 ACM 引用格式，DOI 10.1145/3779212.3790236 未在 ACM DL 打开 †。

- 场景：8 × A100-SXM4-80GB，CUDA 12.8，SGLang 0.3.6；Llama-8B、Llama-70B，均 TP=8；Mooncake 的 Conversation 与 Tool&Agent trace，ShareGPT、LooGLE 合成负载。无硬截止期。
- 目标与指标：P99 TTFT 与 P99 TBT；TBT SLO 8B 为 50 ms、70B 为 100 ms；TTFT SLO 由 Drift 按每 1K 新 token 1 s 设定（基线无 TTFT 约束，比较不对称）；SLO 下峰值吞吐。
- 核心观察：
  - 70B 上 1K 新 token + 127K 缓存 token 的 prefill：KV 迁移 133 ms，重算 15.9 s，直接共享 KV 205 ms。
  - chunked prefill 的延迟甜点是 4K token budget，约 240 ms，超过 <100 ms 的 TBT 目标。
  - prefill 计算需求随复用上下文长度增长，decode 对此不敏感（Figure 3）。
  - 各算子访存 / 计算时间比：Decode Attn 19.2，其余（Extend Attn 0.027、QKV 0.666 等）均 <1（Table 2）。
  - Principle 1：仅当两侧 kernel 都 memory-bound 时争用明显；8600 多个并发组合中 90 分位 slowdown <7 %，最大 17 %；前提是 prefill 至少拿一半 SM。
- challenge：GreenContext 分区带来气泡（prefill launch 慢、decode 结束时间不可预测、负载倾斜）；分区不能频繁重建。
- 机制：预建 4 组 green context（prefill, decode SM）= (108,0)、(84,24)、(72,36)、(0,108)；每组约 4 MB，加上每个分区各捕获一套 decode CUDA Graph 后 8 卡约 743 MB（v2）；prefill 切成 block 级（切分开销 <1.5 %），decode 用 graph 级 launch；用 CUDA event 查询代替阻塞同步；用 solo-run 训练的延迟预测器（最大误差 prefill 8.16 %、decode 8.84 %）给 decode 刚好够用的 SM；prefill 抢占栈深度 1，只在预测总时间不超出 SLO 余量时抢占（Algorithm 1）。
- 基线与评估：SGLang、SGLang chunking（budget 512）、LoongServe、Dynamo。摘要：平均吞吐 5.1×、最多 17.5×；真实 trace 上 P99 TTFT 平均快 2.62× / 7.72× / 9.88×（对 chunking / LoongServe / Dynamo）。
- 与本场景的关系：
  - 成立：同样用 green context、同样只能选少数预建分区；"给延迟敏感方刚好够用的 SM、其余给 prefill"与本场景给执行器留 SM 的思路同构（判断）；block 级切分 prefill 是在不改 forward 数值的前提下缩短不可抢占单元的一种做法（判断，需核对它是否改变数值）。
  - 不成立：争用测量结论"<7 % (P90)"以 108 SM HBM GPU 且 prefill ≥ 一半 SM 为前提；本场景 273 GB/s LPDDR5X、20 SM，规划器 decode 带宽占用已被观测为主要剩余干扰，该结论不能直接迁移；SLO 是 P99 统计量，且受保护方是同一模型的 decode。
- **精读笔记**：[notes/2025-drift.md](notes/2025-drift.md)。卡片与笔记不一致处以笔记为准。

### B14. Nexus

**Nexus** · 2025 · 预印本 · [链接](https://arxiv.org/abs/2507.06608)
Nexus: Proactive Intra-GPU Disaggregation of Prefill and Decode in LLM Serving
Xiaoxiang Shi et al.

- 场景：1–2 × NVIDIA L20-48GB，CUDA 12.8（§6.1）；Qwen2.5-3B、LLaMA3.1-8B、Qwen2.5-14B；Long Data Collections（平均输入 5905）、Arxiv Summarization（3832）、ShareGPT（496）、Mixed（Table 1）。无硬截止期。
- 目标与指标：均值与 P95 的 TTFT、TBT、归一化延迟；吞吐 = 满足 token 延迟约束的最高到达率（约束具体值未给出，§6.2.1）。
- 核心观察：
  - 混合 batch 平均 0.251 s，纯 decode batch 0.015 s；混合 batch 中 linear kernel 最多慢 10×，TBT 升 8× 以上（§3.1, Fig. 4）。
  - 收益递减：prefill SM 从 30 % 到 40 % 延迟降 >25 %，70 % 到 80 % 只降约 10 %；decode 超过 50 % 后每多 10 % SM 收益 <3 %（§3.2, Fig. 5，用 MPS 测）。
  - 带宽争用：prefill KV 长度从 2000 增到 10000，同一 decode batch 延迟升 36 %；prefill 访存流量随时间波动（§3.3, Fig. 6）。
- challenge：分区比例需随负载变化而变，但重配置有代价，且反应式调整会振荡。
- 机制：CUDA Green Contexts（约 150 行 CUDA 的 PyTorch 扩展，§5）；所有分区布局启动时预建；算子级 max(计算, 访存) 成本模型，计算项为饱和-衰减曲线，decode 访存时间用与 prefill attention 重叠时的有效带宽份额（Eqs. 5–9，争用只对 attention 建模）；两阶段贪心搜索 2–4 次模型评估收敛（§4.1.3）；KV 使用 ≤70 % 时 prefill 优先，否则 decode 优先；迟滞阈值防振荡（§4.2）；prefill 用 Shortest-Prompt-First，decode 用 FCFS（§4.3）。
- 基线与评估：vLLM、FastServe（在 vLLM 上重实现）、SGLang、vLLM-P/D（两卡分离）。摘要：对 vLLM 吞吐最多 2.2×、TTFT 低 20×、TBT 低 2.5×；对 SGLang 2× / 2× / 1.7×；以一半 GPU 超过 vLLM-P/D 1.4× 吞吐（§6.2）。消融：仅动态 SM 使 TBT −14 % 但 TTFT +30 %（§6.5, Fig. 13）。
- 与本场景的关系：
  - 成立：显式测量并建模了"decode 被并发 prefill 的访存拖慢"，与本场景"分 SM 后剩余干扰来自带宽"同类；SM 收益递减曲线与本场景"规划器只剩 8 SM 时 L 超 5.33 s"可对照（判断）。
  - 不成立：被拖慢的是 decode、施压的是 prefill；本场景中受害方是执行器，施压方包括规划器 decode；L20 为 GDDR6 独立显卡，带宽量级与统一内存 SoC 不同；SLO 未给具体值。

### B15. Bullet

**Bullet** · 2026 · ASPLOS '26 † · [链接](https://arxiv.org/abs/2504.19516)
Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration（camera-ready 标题；arXiv 页面当前标题为 Boosting LLM Serving through Spatial-Temporal GPU Resource Sharing）
Zejia Lin et al.

† 会议名来自作者主页 camera-ready PDF 首页的 ACM 引用格式（ASPLOS '26, DOI 10.1145/3779212.3790135）；未在 ACM DL 或 DBLP 核对。arXiv v4 标题改为 "Boosting LLM Serving through Spatial-Temporal GPU Resource Sharing"。

- 场景：8 × A100-80GB、8 × H100、8 × H20（Table 3）；Llama3.1-8B/70B、Qwen3-235B-A22B-FP8；ShareGPT、Azure-Code、arXiv-Summary，Poisson 到达。无硬截止期。
- 目标与指标：吞吐；归一化 TTFT / TPOT SLO（Table 4，例 ShareGPT 3.0 ms / 150 ms）；P90 SLO attainment（Fig. 14d）。
- 核心观察：
  - wave quantization：1024 token 时理论 SM 空闲比整层 19.4 %、O-proj 40.7 %；16384 token 时整层 0.5 %（§2.2.1, Table 1）。
  - 整层实测计算利用率只有 70–76 %（§2.2.2, Fig. 2）。
  - prefill 随 SM 数近线性，decode 超线性（§2.2.3, Fig. 3，仅图）。
  - 争用：up-gated GEMM 在 >60 % SM 时退化 <8 %；内存拷贝吞吐随并发 prefill 序列长度上升而下降（§3.2.3）。
  - chunked prefill（1k 块）：计算效率从 71 % 降到 61 %，最后一块是第一块的 1.9×，总 prefill 延迟为不切块的 1.13×（§2.3.2, Fig. 5）。
- challenge：分区要随请求动态变化，green context 预建开销大（文中引 Drift：4 组 >700 MB）；需要能预测并发时延迟。
- 机制：MPS 开启下用 libsmctrl 对每个引擎的 stream 设 SM mask，重配置平均 4.1 µs（Table 5）；SM 缩放 roofline + 校准因子 + 内存子系统争用项的性能模型（Eq. 1–2；MAPE prefill 9.1 %、decode 10.5 %，Table 2；单次预测 10.2 µs）；层级 prefill 调度、decode 走 CUDA Graph；按 P90 TTFT / TPOT 重分 SM（§3.3, Alg. 1）；权重与 KV 用 cudaIpc 共享，decode 接收 prefill 结果无需传 KV。
- 基线与评估：vLLM v0.8.5、SGLang v0.4.6（1024 / 2048 块）、Nanoflow。摘要：平均吞吐 1.26×、最多 1.55×；单 GPU 对 SGLang-1024 吞吐 1.09×（§4.2.1）；对 Nanoflow TTFT 长 2.37×、吞吐低 20.0 %（§4.2.2）；Nsight 测得 active SM cycles 86.2 %（§4.3.2）。
- 与本场景的关系：
  - 成立：MPS + 每 stream SM mask、微秒级切换，提供了 MPS 百分比（需重启）与 green context（预建、8 SM 粒度）之外的第三种分区手段；性能模型显式含争用项，可作为预测执行器在并发下延迟的参考形式（判断）；层级 prefill 调度给出比整块 forward 更细的让出点（判断）。
  - 不成立：libsmctrl 依赖未公开的驱动接口，在 Thor（Blackwell, JetPack）上是否可用未知；作者自述在低算力 GPU 上 decode 需要更大 SM 份额才能吃满带宽，增益会缩小（§4.6），Thor 正属于低 SM 数情形；SLO 为统计量，无跨模型截止期。

### B16. DuetServe

**DuetServe** · 2026 · ICML 2026（PMLR v306）· [链接](https://proceedings.mlr.press/v306/gao26m.html)
DuetServe: Harmonizing Prefill and Decode for LLM Serving via Adaptive GPU Multiplexing
Lei Gao et al.

- 场景：1 台服务器 2 × H100-80GB（NVLink），CUDA 13.0（§5.1）；Qwen3-8B（TP=1）、Qwen3-14B（TP=2），附录 Qwen3-32B TP=8；Azure-Code（ISL 2047 / OSL 28）、Azure-Conv（1155 / 211）、Mooncake（12035 / 343）（Table 1）。无硬截止期。
- 目标与指标：在 TBT 上界 τ_TBT 约束下最大化吞吐（Alg. 1）；正文引 100 ms 为典型 TBT SLO，评估以 150 / 200 ms 为参照线。
- 核心观察：
  - linear 层饱和点从 A100 约 2K token 移到 H100 约 8K；8192 token 的纯 prefill batch "consistently exceed 180 ms"（§3, Fig. 1b）。
  - 8B、2×H100、8000 in / 200 out 时，1P+1D 分离的吞吐不到聚合 vLLM 的一半（§3, Fig. 2）。
  - HBM 带宽随活跃 TPC 超线性增长，约 20 % SM 可达约 60 % 峰值带宽；FLOPs 近线性（§3, Fig. 3a）。
- challenge：分区本身有代价，只在需要时开启；分区后 decode 每步若都与 CPU 同步，开销抵消收益。
- 机制：先按 chunked prefill 组 batch，roofline 模型预测混合 batch 延迟，超出 TBT 上界才开启空间复用（§4）；libsmctrl 每 stream SM mask，TPC（2 SM）粒度，不用 MPS（单进程不可用）与 GreenContext（需预建）（§2）；分区搜索以 2 SM 步长枚举 decode SM 数，保证预测 decode 延迟 ≤ τ_TBT（§4.2）；look-ahead decode 预分配 k 步 KV 槽并连续发 k 个 CUDA Graph，graph launch <0.5 ms（§4.3）。
- 基线与评估：vLLM v0.10.1、SGLang v0.5.0（默认 / chunked）、Dynamo v0.5.1（1P+1D）。摘要：吞吐最多 1.3×；Azure-Code QPS 16 时 13.57 vs 12.43 req/s，TBT 保持 <150 ms，而 vLLM 与 SGLang-Chunked >200 ms（§5）；8×H100 上对 Dynamo 约 1.4×，但 TBT 104.7 ms vs 23.1 ms（Table 3）。轻载时 TTFT 略高（§5.2）。
- 与本场景的关系：
  - 成立："约 20 % SM 即可拿到约 60 % 带宽"的测量说明少量 SM 的 decode 就能占用大部分带宽，与本场景"分 SM 后 decode 带宽仍干扰执行器"方向一致（判断，硬件不同）；"按需开启分区"对应本场景只在 prefill 与执行器重叠时才需要隔离（判断）。
  - 不成立：并发 prefill 与 decode 之间的 HBM 争用未分析；单进程两 stream，本场景为两进程两模型；libsmctrl 在 Thor 上可用性未知。

## C. 边缘 SoC 上的多模型实时推理与内存带宽争用

### C17. Miriam

**Miriam** · 2023 · SenSys · [链接](https://doi.org/10.1145/3625687.3625789)（预印本 [arXiv:2307.04339](https://arxiv.org/abs/2307.04339)）
Miriam: Exploiting Elastic Kernels for Real-time Multi-DNN Inference on Edge GPU
Zhihe Zhao et al.

- 场景：NVIDIA GeForce RTX 2060 与 Jetson AGX Xavier（§8.1.1，CUDA 11.2）。负载为自建 MDTB（A–D 四组，6 个 CV/NLP 模型，如 AlexNet、CifarNet、ResNet、SqueezeNet），分关键任务与普通任务；另用 LGSVL 自动驾驶 trace：关键任务 ResNet 10 Hz、普通任务 SqueezeNet 12.5 Hz（评估节，Fig.12，仅在 2060 上）。没有显式截止期数值，关键任务以“延迟接近独占”为目标。
- 目标与指标：关键任务端到端延迟尽量接近 Sequential（独占），同时最大化整体吞吐；指标为关键任务延迟、整体吞吐、achieved occupancy（§8.1.4）。
- 核心观察：
  - 同时运行时 ResNet50 延迟分布明显变宽（Fig.2 左，仅图）；把争用分成 intra-SM（同一 SM 内寄存器、shared memory、执行单元）与 inter-SM（全局内存、内存控制器）两类（背景 / 动机节，Fig.2 右）。
  - Tango benchmark 中只有 7.4% 的 kernel 可以直接改 grid/block 而不改计算调度（elastic kernel 一节），说明不能简单改 launch 参数。
  - 作者在引言写明 MPS 与 MIG 在边缘 GPU 上不可用。
- challenge：关键任务优先与整体吞吐两个目标冲突；边缘 GPU 缺硬件级资源管理；elastic kernel 的可选 schedule 数量随算子数指数增长（AlexNet 全部 CONV 可达 2.2×10²⁵ 种，elastic kernel 生成一节）。
- 机制：离线把普通任务的 kernel 经 source-to-source 变换成 grid/block 可调的 elastic kernel，用 WIScore 等约束剪枝候选（剪掉 84%–95.2%，Fig.10）；在线 coordinator 在关键 kernel 到来时，只用关键 kernel 剩下的 SM 资源去“填充”普通任务的 elastic kernel 分片（shaded binary tree，贪心选择，Fig.7，运行时协调一节）。
- 基线与评估：Sequential、GPU Multi-stream with Priority（Triton 方式）、Inter-stream Barrier。MDTB A 上，相对 Sequential，Multi-stream 与 IB 让关键任务延迟增加 1.95× / 1.52×（2060）和 2.02× / 1.77×（Xavier），Miriam 只增加 21% / 28%，吞吐提升 64% / 83%（Fig.8）。LGSVL trace：吞吐 +89%，关键任务延迟 +11%（Fig.11）。调度开销小于 0.35 ms；超过 80% 的情况下关键 kernel 的 launch 额外开销小于 15 µs，作者将其归因于 texture cache 与 L2 争用，并留作未来工作（评估节的开销分析）。在 Xavier 上作者指出其内存带宽较低限制了收益（Fig.8 的讨论）。
- 与本场景的关系：
  - 成立：同一块 GPU 上一个关键任务加一个尽力而为任务；关键任务以“接近独占延迟”为目标，与执行器的诉求一致。
  - 不成立：Thor 上有 MPS 与 green context，论文“MPS 不可用”的前提不成立；普通任务以吞吐为目标，而本场景的规划器有自己的延迟要求（L ≤ 5.33 s）；负载是小 CNN，而规划器是 27B VLM 的 prefill / decode，kernel 多来自厂商库或手写融合 kernel，source-to-source 变换是否可行未知（判断）；论文控制的是 SM 级资源，不控制内存带宽，而本场景按 SM 划分后剩余的干扰正来自 decode 的带宽。

### C18. RT-mDL

**RT-mDL** · 2021 · SenSys · [链接](https://doi.org/10.1145/3485730.3485938)
RT-mDL: Supporting Real-Time Mixed Deep Learning Tasks on Edge Platforms
Neiwen Ling et al.

- 场景：Jetson AGX Xavier（主平台）、TX2、Nano，以及 i9 + RTX 2080 桌面机；F1/10 自动驾驶小车案例（Section 6）。负载为周期任务（AlexNet、VGG11/13/16/19、ResNet18/34、LeNet 等，以及 SBCNN、LSTM），截止期设为平均执行时间的 4 倍（紧）或 6 倍（松）（Section 6，Fig.17）。
- 目标与指标：在每个任务的截止期错过率上界（如 5%/10%/20%）约束下，最小化归一化精度损失之和（Eq.1）；指标为截止期错过率与精度损失比。
- 核心观察（Section 3，在 Xavier / TX2 / 桌面机上 profiling）：
  - 所有优先级分配下，最差任务的错过率仍为 81.39%（桌面）、99.33%（Xavier）、72.40%（TX2）（Fig.3），即只调优先级不够。
  - 前后处理在 CPU 上约占任务时延的 30%（Fig.5），导致 CPU 与 GPU 的时间利用率都不高。
  - AlexNet、VGG11 等单个模型占不满 GPU，空间浪费 14.54%–36.25%（Fig.6）。
  - VGG11 精度降 1%（81.64%→80.34%）延迟降 27.65%（10.23→7.40 ms）（Fig.2，Xavier）。
- challenge：错过率没有闭式表达；模型变体占存储（VGG19 9 级变体共 605.6 MB）；主流 DL 框架和加速器缺少实时优先级支持。
- 机制：存储受限的多级模型缩放（权重共享）；按优先级的调度器，把任务拆成 CPU 与 GPU 子任务放进各自的优先级队列（mutex 实现），并做 GPU packing；用 MOEA 搜索“模型变体 + 优先级”，错过率由 Random Forest 代理模型预测（误差低于 10%，Fig.9）。
- 基线与评估：RTA 方法，DM-NMC、DM-MVS、DM-MC、SPO-NMC、UM-NMC。轻负载下 AlexNet-2 错过率 6.15%，对比 SPO-NMC 8.94%、UM-NMC 16.29%（Fig.17a）；Nano 上精度损失 1.17%，错过率比 UM / DM 分别低 8.79% / 5.41%（Fig.18）。摘要称错过率平均降低 40.12%、精度损失 1.7%（仅摘要）。
- 与本场景的关系：
  - 成立：周期任务、截止期与错过率指标、CPU 与 GPU 共存。
  - 不成立：主要控制量是“降模型精度换延迟”，本场景中规划器改变分块已会改变输出，模型缩放大概率不可接受（判断）；GPU 子任务通过 mutex 串行，不存在两个模型在 GPU 上同时运行的争用，而本场景的核心现象就是并发争用；任务规模为毫秒级 CNN，没有秒级 prefill 这样的长非抢占段。

### C19. Twill

**Twill** · 2025 · ICCAD · [链接](https://doi.org/10.1109/ICCAD66269.2025.11240767)（预印本 [arXiv:2507.00491](https://arxiv.org/abs/2507.00491)）
Twill: Scheduling Compound AI Systems on Heterogeneous Mobile Edge Platforms
Zain Taufique et al.

- 场景：Jetson Orin NX（GPU + DLA）（§IV-A）。负载为 compound AI 任务组合 Mix 1–5：DNN（VGG-19、ResNet-152、EfficientNet-b4）、encoder transformer（Bert-base/large、ViT-base），Mix 4–5 含 LLM；请求到达时间运行时未知。没有截止期，只有功耗预算。
- 目标与指标：在功耗预算内最小化推理延迟；指标为各请求延迟、总执行时间与运行时功耗（Fig.6、Fig.7）。LLM 以 100 个输出 token 的延迟计（Fig.9）。
- 核心观察：现有边缘方法多针对 multi-DNN 或仅 transformer，依赖设计期 profiling；transformer 算子大多不被 DLA 支持，只能在 GPU 上跑（引言与 Fig.5 的 DLA 兼容性实测）；DNN 与 transformer 同时在 GPU 上会因共享资源争用而延迟都变高（Fig.3 示例）。
- challenge：请求到达顺序未知；模型对 GPU / DLA 的亲和度不同；需要同时管功耗。
- 机制：Model Interpreter 在线检查每层与 DLA 的兼容性（Algorithm 1）；Controller 以 Analyze–Decide–Deploy 循环执行四种手段：亲和度感知的 cluster 映射、cluster 间迁移（切分层）、按优先级冻结 / 解冻任务、DVFS。每个 cluster 同一时间最多映射一个模型（框架设计节，Fig.4）。
- 基线与评估：Tango、Mapformer、Band 等三种已有方法（Section IV）；摘要报告平均延迟降低 54%（仅摘要），同时满足功耗预算。
- 与本场景的关系：
  - 成立：同一 Jetson SoC 上同时存在 DNN 与 LLM。
  - 不成立：没有截止期，目标是平均延迟与功耗；它通过“一个 cluster 只跑一个模型 + 冻结低优先级任务”来回避 GPU 上的并发，而本场景两个模型都只能用 GPU（可用控制中没有 DLA），争用无法靠换加速器回避；冻结相当于任务级抢占，本场景规划器 prefill 一块约 1.17 s，任务级冻结的粒度与执行器 200 ms 截止期不匹配（判断）。

### C20. Edge GPU 隔离对比

**Edge GPU 隔离对比** · 2026 · 预印本 · [链接](https://arxiv.org/abs/2601.07600)
Peformance Isolation for Inference Processes in Edge GPU Systems（标题原文拼写如此）
Juan José Martín et al.

- 场景：NVIDIA A100（MPS、MIG）与 Jetson Orin Nano、Jetson Orin AGX（MPS、Green Contexts）（§IV-A）。负载为 ConvNeXt Base/Large、ViT B16/L32、MobileNetV2、ResNet18，batch 1，面向安全关键的多模型集成（ensemble）推理。
- 目标与指标：给定隔离机制下，一个进程以其最大稳定推理频率（IMS，Algorithm 1 求出）运行时，另一进程逐步提高推理频率，统计前者超时（timeout）推理的百分比（Fig.8–11）；另测吞吐、显存与功耗随分区大小的变化（Fig.4、Fig.5）。
- 核心观察：
  - Orin Nano 上 MPS 的 IMS 与独占相同（如 ConvNeXt Large 33 / 33，MobileNetV2 134 / 134），GC 4 SM 分区降为 18 与 79（Table III）。
  - Orin Nano 在两个进程并发时受功耗限制，GPU 频率下降，使 GC 的隔离变差（Fig.6，Fig.9）。
  - 在 Orin AGX 上把 GPU 频率固定为 1.02 GHz 后，GC 的超时比例大幅下降，接近 A100 上 MIG 的隔离水平（Fig.10 及其所在的 Orin AGX 两进程实验小节）。
  - 大模型偏 compute-bound，SM 减少时性能缓慢下降；小模型偏 memory-bound（最大 IMS 结果小节）。
  - GC 只隔离 SM，不隔离内存，这是其无法提供完整时间保证的原因（结论节）。
- challenge：MIG 只在数据中心 GPU 上有；GC 无内存隔离；边缘板上功耗与频率耦合。
- 机制：本文是测量对比，不提出新机制。
- 基线与评估：GPU stand-alone（无隔离并发）、MPS、MIG（A100）、GC（Jetson）；2 进程与 4 进程实验。
- 与本场景的关系：
  - 成立：Jetson 平台、MPS 与 green context 正是本场景可用的控制手段；“GC 不隔离内存带宽”的结论与本场景“按 SM 划分后剩余干扰来自 decode 带宽”一致。
  - 不成立：两个进程运行同一个小视觉模型，角色对称；没有 LLM、没有长 prefill、没有 CPU 内存负载；没有评估 MPS 客户端优先级；Orin AGX 的结果是在固定频率下得到的。

### C21. BWLOCK++

**BWLOCK++** · 2018 · ECRTS · [链接](https://doi.org/10.4230/LIPIcs.ECRTS.2018.19)（预印本 [arXiv:1712.08738](https://arxiv.org/abs/1712.08738)）
Protecting Real-Time GPU Kernels on Integrated CPU-GPU SoC Platforms
Waqar Ali et al.

- 场景：Jetson TX2，Linux 4.4.38，CUDA 8.0，固定最高频率（评估节）。实时 GPU 任务为 Parboil benchmark（histo、SAD 等），CPU 干扰为 IsolBench 的 Bandwidth（3 个实例，顺序更新大数组）。
- 目标与指标：在内存密集的 CPU 任务干扰下保护实时 GPU kernel 的执行时间，同时尽量少损失 CPU 吞吐；指标为 GPU benchmark 总执行时间的 slowdown 与系统总 throttle 时间。
- 核心观察：3 个 Bandwidth 共跑时，histo 最坏 slowdown 超过 250%，SAD 超过 150%（Fig.7；摘要称最坏约 3×）；在 Linux CFS 下，被限流的内存密集任务 vruntime 增长慢，反而被更多调度，加剧限流造成的 CPU 浪费（§4.3，Fig.5–6）。
- challenge：阈值依赖每个 GPU 应用的敏感度；CPU 被限流会损失吞吐；需要不改程序就知道 kernel 的起止时刻。
- 机制：在动态链接层拦截 CUDA API，kernel 执行或 memcpy 期间自动持有“内存带宽锁”（按 stream 跟踪）；持锁时由 per-core 带宽调节器（PMC 计 LLC miss，1 ms 周期，超额即 throttle 该核）限制 best-effort CPU 核；Throttle Fair Scheduler（TFS）把被限流时长乘惩罚系数加回 vruntime。每个 GPU 应用的安全带宽阈值由“slowdown 随共跑带宽变化”的 profiling 曲线确定（Fig.8）。
- 基线与评估：Solo、Corun、BW-Locked-Auto；调度对比 CFS、TFS-1、TFS-3。TFS 相对 CFS 平均减少系统 throttle 时间 39%（TFS-1）和 62%（TFS-3）（Fig.10）。
- 与本场景的关系：
  - 成立：统一内存 SoC 上 CPU 内存负载拖慢 GPU，与本场景“CPU 内存负载让 decode 慢约 2.2 倍、执行器 p99 翻倍”同一类现象。
  - 不成立：假设所有实时任务在一个专用 CPU 核上、只有实时任务使用 GPU、GPU kernel 串行（论文的 limitations 段）；本场景两个模型都在 GPU 上，GPU 内部（规划器对执行器）的带宽争用不在其模型之内；在 Thor 的 Arm 核上用 PMC 做 1 ms 级 per-core 限流是否可行未核实（判断）。

### C22. 动态带宽分配

**动态带宽分配** · 2020 · IEEE TCAD · [链接](https://doi.org/10.1109/TCAD.2020.3012210)（全文 [NSF PAR](https://par.nsf.gov/servlets/purl/10267567)）
Dynamic Memory Bandwidth Allocation for Real-Time GPU-Based SoC Platforms
Homa Aghilinasab et al.

- 场景：Jetson TX2；实时任务为 Parboil GPU kernel；best-effort CPU 任务为 bandwidth 合成负载与 SPEC（462.libquantum 高、403.gcc 中、458.sjeng 低内存强度）（Section VII）。3 个 A57 核最大合计带宽约 3.9 GB/s，对应每 1 ms 约 60,000 次 LLC miss（Section VII）。
- 目标与指标：在不超过 GPU kernel 执行预算的前提下，尽量多给 best-effort CPU 任务带宽；指标为 WCET 估计偏差（Table II）与 best-effort 任务发出的内存请求数相对 NOMINAL 的提升（Table III、IV）。
- 核心观察：静态限流因 WCET 估计悲观而过度压制 CPU；GPU kernel 的 thread block 执行路径少，可按执行时间分布聚类，用“各类剩余 block 数”衡量进度（Section IV-A）；6 个内存密集 benchmark 的名义预算低于 30%，5 个计算密集的在 Q=1 时 slowdown 都低于 5%，动态分配对后者无用（Section VII）。
- challenge：GPU 内部调度细节不公开，无法做静态 WCET 分析；BWLOCK++ 只限制每周期访问次数、不限制时刻，最坏访存模式难界定。
- 机制：离线测每个 thread block 的执行时间分布并用 K-S 检验聚类；建立以带宽比例 Q 为参数的测量式 WCET 模型（Algorithm 1，LP 求解）；运行时由用户态高优先级进程读取已完成 block，按剩余工作重算本周期可给 CPU 的带宽（FAIR / GREEDY / SMOOTH，平滑系数 0.3），通过 BWLOCK++ 内核模块执行（Section V、VI）。
- 基线与评估：NOMINAL（静态预算）对比 FAIR / GREEDY / SMOOTH。合成负载下 best-effort 内存请求数提升 74%–195%（libquantum 行）、21%–42%（gcc 行）、0%–13%（sjeng 行）（Table III）。
- 与本场景的关系：
  - 成立：“实时任务有余量时把带宽让给尽力而为任务”的思路，与执行器单独 37.5 ms、截止期 200 ms 之间的余量相对应（判断）。
  - 不成立：同样只处理 CPU→GPU 方向，假设 GPU 一次只运行一个 kernel、不可抢占（Section III）；本场景的尽力而为方是 GPU 上的规划器，不是 CPU 任务，现有手段中没有对 GPU 进程按带宽限流的接口。

### C23. Serafini 博士论文第 6 章

**Serafini 博士论文（第 6 章）** · 2026 · 博士论文（UNIMORE；IRIS 记录的 issued 日期 2026-03-31；第 6 章没有同行评审版本）· [链接](https://iris.unimore.it/handle/11380/1403093)
Demystifying Memory Interference: Analysis and Mitigation of the Memory Contention Problem in Heterogeneous Systems-on-Chip
Andrea Serafini

- 场景：Jetson AGX Orin（Jetson Linux 36.4.4）与 Jetson AGX Thor（Jetson Linux 38.2），关闭 CPU / GPU / 内存 DVFS（第 6 章实验设置节）。CPU 侧为关键内存密集任务 cpu_synth；GPU 侧为合成内存流量生成器（16 B 访存，read / write / readwrite，stride 16 B 或 512 B）与 PolyBench-ACC kernel（Section 6.4、6.5）。
- 目标与指标：保护 CPU 关键任务不受 GPU 引起的内存干扰；指标为 CPU 访存延迟 slowdown、GPU 带宽与 GPU kernel slowdown。
- 核心观察：
  - 合并访存时 GPU 可接近打满带宽：Orin 192 GB/s（标称的 93.6%），Thor 257 GB/s（标称的 94.1%）（Fig.6.9c、Fig.6.10a）。这些带宽是内存控制器 actmon 活跃比例 × 标称带宽，不是 GPU 侧实测吞吐。
  - 合并写流量对 CPU 的 slowdown：Orin 1.55×，Thor 2.5×（Fig.6.9b、Fig.6.10b）。
  - 512 B 跨步访问在 Thor 上更严重：正文写读 3×、写 5×、读写 3.1×，图中约 2.9× / 4.5×（Fig.6.12）；Thor 上跨步写即使只用 1 个 SM 也能压满内存控制器，对 SM 限制不敏感（Section 6.4 及第 1 章贡献概述）。
- challenge：已有 GPU→CPU 方法多在时间域（PREM 需改代码，利用 GPU 空闲周期的方法粒度粗）；MIG 分区选择少，MPS 不控制 SM 分配。
- 机制：空间域带宽管理：根据 CPU 任务声明的 QoS（0–100）用 Green Contexts 动态限制 GPU 可用 SM 数，从而降低 GPU 带宽需求（第 6 章 MBMS 设计节）。论文写明 GC 最小分区为 2 SM、每档增加 2 SM（§6.2.2、§6.3.2），未给 CUDA 版本，也没有测 green context 实际使用的 SM；CUDA Driver API 文档（v13.5）对 Tegra 写的也是“multiple of 2”。本组在 Thor 上测得请求 4 SM 实得 8 SM，两者不一致，原因未核实。
- 基线与评估：不调节（全 SM）对比不同 GC 分区，以及用自写 kernel 模拟 1 SM 的“理想 GC”。合并写流量下，GC 调节把 slowdown 从 1.55× 降到 1.2×（Orin）、从 2.5× 降到 1.6×（Thor），模拟 1 SM 时为 1.15× / 1.2×；Thor 跨步读从 3× 降到 1.4×，跨步写不受调节影响（Section 6.4，Fig.6.9–6.12）。PolyBench-ACC 结果以热力图给出（Fig.6.13–6.16）。
- 与本场景的关系：
  - 成立：同一块 Jetson AGX Thor，同样的 273 GB/s 统一内存与 green context；直接说明“限 SM 只能部分限制带宽，某些访存模式下 1 个 SM 就能压满内存控制器”，与本场景“按 SM 划分后 decode 带宽仍造成干扰”的现象方向一致（判断）。
  - 不成立：受保护方是 CPU 任务，本场景受保护方是 GPU 上的执行器，干扰方是 GPU 上的规划器（GPU 对 GPU）以及 CPU 内存负载（CPU 对 GPU）；负载是合成流量与 PolyBench，没有 DNN / LLM；学位论文，未经同行评审的会议版本未核对到。
- **精读笔记**：[notes/2026-serafini-thesis-ch6.md](notes/2026-serafini-thesis-ch6.md)。卡片与笔记不一致处以笔记为准。

## D. 机器人双系统的部署

### D24. TRT-Edge-LLM PR #193

**TRT-Edge-LLM PR #193** · 2026 · GitHub PR（NVIDIA/TensorRT-Edge-LLM，2026-08-24 提交，状态 open，未合并）· [github.com/NVIDIA/TensorRT-Edge-LLM/pull/193](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/193)
feat: add InternVLA-N1-DualVLN vision-language navigation support
hungho77（GitHub 账号）

- 场景：Jetson Thor（sm_110，JetPack 7.1，TRT 10.13.3.9，CUDA 13），batch 1。System 2 = Qwen2.5-VL-7B 规划器（FP16 / FP8 / NVFP4 TensorRT 引擎，或 PyTorch bf16），输出 `z_latents`；System 1 = flow-matching 轨迹专家 `traj_dit`（10 步 × 32 条样本）+ DINOv2 记忆块，全部 BF16。规划节奏 `--cadence 4`（每 4 个 System 1 tick 请求一次规划）。没有硬截止期，用 control rate 和最坏 tick 表示。
- 目标与指标：System 1 的 control rate（Hz）/ 每 tick 平均与最坏延迟；System 2 prefill（1024 token）与 decode（pastKV 1024）延迟；首个规划延迟；引擎大小；`z_latents` 与轨迹的 cosine；闭环 R2R val_unseen 成功率（199 episode，McNemar 检验）。
- 核心观察（均出自 PR 正文 Validation 节）：
  - 共置干扰表："System 2 idle" 每条轨迹 48.1 ms（20.8 Hz）；"System 2 running, same process + priority stream" 55.9 ms（17.9 Hz）；"System 2 running, separate process" 117–122 ms（8.2–8.5 Hz）。作者解释："CUDA orders streams within a context, so across processes the GPU merely time-slices"。
  - System 1 本身不变（约 52 ms），只量化 System 2 时 control rate 从 FP16 的 15.0 Hz 到 NVFP4 的 18.0 Hz；作者归因为"how much the planner crowds it"（Latency and control rate 表及其后文字）。
  - 异步与阻塞对比：每 tick 55.9 对 57.9 ms，最坏 tick 78 对 331 ms（PyTorch 参考实现在 System 2 规划时阻塞）。
  - commit 记录里的竞争实验（非 PR 正文）：两个轨迹循环并发各 104 ms（单独 47 ms）；流优先级下单独 48.2 ms、同优先级 94.0 ms、高优先级 73.3 ms；跨进程加 control stream 仍 121 ms。
- challenge：同一 GPU 上规划器与执行器必须重叠；跨进程时流优先级失效；核心运行时按"组件不重叠"取 max 来分配共享上下文内存池，与重叠执行冲突。
- 机制：单进程；System 1 跑在高优先级 CUDA stream（`makeControlStream()`）；规划器线程异步运行，请求合并，System 1 始终读最新规划并报告 staleness；System 1 用独立的上下文内存池。不用 MPS，不分 SM。
- 基线与评估：PyTorch bf16 参考实现（阻塞式）；FP16 / FP8 / NVFP4 / NVFP4+QAD 引擎；闭环 SR 无显著差异（例如 NVFP4 67.8% 对 PyTorch 69.8%，p = 0.572）。
- 与本场景的关系：
  - 成立：同一块 Thor；规划器 7B VLM 与动作模型同时运行；规划器异步、执行器读最新规划；观察到规划器越"挤"执行器越慢。
  - 不成立：规划器只 prefill 1024 token、约 75 ms，远小于本场景 4–6k token、1.2–2.2 s 的 prefill；执行器无严格截止期，也没有报告 p99 或错过率；只比较了"同进程 + 流优先级"与"跨进程时间片"，没有测 MPS 客户端优先级、SM 划分、green context，也没有带宽或 CPU 内存负载数据。
  - 判断：它"跨进程只能时间片"的结论是在不开 MPS 的条件下得出的，与本场景"MPS 客户端优先级有效"不矛盾。

### D25. DualVLN on Thor

**DualVLN on Thor** · 2026 · NeurIPS 2026 Workshop（On-Device Intelligence: Foundation Models under Real-World Constraints，poster）· [OpenReview Ei1Wgdo2I0](https://openreview.net/forum?id=Ei1Wgdo2I0)，[neurips.cc](https://neurips.cc/virtual/2026/170076)
DualVLN on Jetson Thor: Measuring Inference Cost, Observation Freshness, and Navigation Quality
Yoojin Lim et al.

- 场景：单块 Jetson Thor，不重训，把 DualVLN（7B 规划器 System 2 + diffusion 轨迹模型 System 1）移植为 TensorRT + NVFP4 规划器；H1 人形机器人在物理仿真中导航；仿真在模型计算期间冻结场景（摘要）。对照的原 DualVLN（Wei et al.，ICLR 2026，[arXiv 2512.08186](https://arxiv.org/abs/2512.08186)）在远端 RTX 4090 上跑全模型，System 2 2 Hz、System 1 30 Hz（图 1 说明），System 2 KV 复用后 1.1 → 0.7 s，System 1 用 TensorRT 0.03 s 生成 32 条轨迹（§5.2），两系统异步（§3.2）。
- 目标与指标：中位延迟（规划、look-down 规划、轨迹生成三处）、峰值 RAM、当前观测的 p95 年龄、System 1 service rate、成功率。
- 核心观察（仅摘要）：中位延迟降 约 6.8×（规划）、7.9×（look-down 规划）、3.9×（轨迹生成），"depending on where timing is measured"；峰值 RAM（高于 idle 的部分）−58.5%；12 条选择路径上成功率 0.556 → 0.458；四种运行时设置中，每个 System 1 输出只用一次（而不是用到 4 个动作）使 p95 观测年龄 2.91 → 0.99 s，System 1 service rate 相近；另 12 条路径各 48 次运行，原实现 / 移植 / 最终设置成功 20 / 23 / 22 次；结论 "inference speed alone determines neither observation freshness nor navigation quality"。
- challenge：推理变快不一定带来更新鲜的观测和更好的导航；小样本下的设置选择会过拟合。
- 机制：TensorRT 移植 + NVFP4 规划器 + 运行时设置（System 1 输出的复用次数）。两模型在 Thor 上的进程 / 流结构摘要未写 †。
- 基线与评估：同设备上原 PyTorch 执行；12 + 12 条路径，各 48 次运行。
- 与本场景的关系：
  - 成立：单块 Thor、同时承载 7B 规划器与动作模型；把"观测年龄"而不是吞吐作为指标，与本场景关注的规划延迟 L 同类。
  - 不成立 / 未知：仿真冻结场景，延迟不影响世界演化；没有截止期；全文未拿到，是否测了两模型间干扰未知 †。

### D26. NavGPT-3

**NavGPT-3** · 2026 · 预印本 · [arXiv 2610.10787](https://arxiv.org/abs/2610.10787)
NavGPT-3: Harnessing Context in a Hierarchical Navigation Runtime
Gengze Zhou et al.

- 场景：仿真用 8B VLA，规划器是商用模型（GPT-6 Astra 经 Codex、Claude Opus 5 经 Claude Code，§4）；真机 Go2：4B VLA 在 Jetson Thor 上 FP8 + TensorRT（§4.8.1），规划器 "the hosted Planner remains a separate service"（§4.8.1），即不在机器人上。LiDAR 安全监视器属于运行时。
- 目标与指标：仿真 SR / 每 episode 时间；真机 repair time、stale command 次数、触发到停车的 halt latency p95（表 10）。
- 核心观察：VLA 步间隔 0.78 s，1–2 Hz（表 3）；规划器每回合（不含 VLA）约 3.3–19.3 s（表 3，§4.2）；真机 repair time：规划器 + 工具 14.2 s，顺序/排队 12.8 s，重叠/优先级 7.9 s（表 10A）；halt latency p95：规划器 + 工具 2150 ms，顺序/排队 1840 ms，顺序/优先级 420 ms，重叠/排队 1310 ms，重叠/优先级 180 ms（表 10B）。
- challenge：规划器秒级、VLA 亚秒级、安全监视需要百毫秒级抢占，三者要并行且谁能驱动机器人必须明确。
- 机制：推理、执行、监视各为一个线程，有各自上下文和权限；运行时按优先级规则转移 "motion authority"（§3.1.3）；真机做 2×2 消融：规划器与 VLA 顺序 / 重叠 × 监视器排队 / 优先级（§4.8.3）。
- 基线与评估：规划器 + 工具（无 harness）；四种 2×2 组合；各 30 次（papers.md B1 记录）。
- 与本场景的关系：
  - 成立：规划器与执行器异步重叠；把抢占延迟作为指标。
  - 不成立：规划器在云端，与 VLA 不共用计算资源，所以"重叠"不产生 GPU / 带宽干扰；VLA 只 1–2 Hz；判断：表 10 的收益来自软件层的调度（谁先拿到控制权），不是设备上的资源隔离。

### D27. OxyGen

**OxyGen** · 2026 · NeurIPS 2026（主会 poster；arXiv comment、PDF 首页脚注、neurips.cc poster 149556 三处一致）· [arXiv 2603.14371](https://arxiv.org/abs/2603.14371)
OxyGen: Unified KV Cache Management for VLA Inference under Multi-Task Parallelism
Xiangyu Li et al.

- 场景：RTX 4090 与 Jetson AGX Thor（§4.1.1）；人形机器人板载 Thor（§4.7）。π0.5（MoT：共享 VLM prefill + 动作专家，S = 10 去噪步，H = 10 + 自回归语言专家），每帧出动作块，同时有跨多帧的语言生成（记忆、旁白、QA）。动作有每帧截止期以维持目标频率 f_min，语言是软约束（v5 §3.1；v2 写作 hard per-frame deadline）。
- 目标与指标：最大化语言吞吐 τ = Bk/T，约束 action frequency f = H/T ≥ f_min（式 3）；另报峰值内存、功耗、每请求能耗。没有 tail latency 或 deadline miss（可见正文中）。
- 核心观察：4090 上只做 KV 共享 200.3 → 145.1 ms（1.4×，N = 5）；加跨帧批处理 469.3 → 175.8 ms，21.3 → 56.9 Hz（N = 30，k = 5）（§4.3）；持续动作频率 4090 约 60 Hz、Thor 约 27 Hz（§4.3）；Thor 上 MPS 并行反而降性能（§4.2）；Thor 真机 isolated 1030 ms（207.5 + 822.3）→ 393 ms（198.0 + 195.4），语言解码 195 ms 藏在 333 ms 的动作执行内（§4.7）。
- challenge：多任务各自 prefill 同一观测造成重复计算；变长语言解码与定频动作生成争资源。
- 机制：同一模型内统一 KV 管理：跨任务共享 prefill KV；跨帧 continuous batching，每帧让每个语言请求前进 k 个 token（算法 1）。不是两个模型的共置。
- 基线与评估：顺序隔离执行（openpi 风格）；并行隔离执行（同 GPU 多进程 + CUDA MPS）；另两种骨干（表 1）；LIBERO 成功率 96.76% 对 96.85%（§4.6）。
- 与本场景的关系：
  - 成立：Thor 上；动作有每帧约束、语言是软约束，目标形式（约束下最大化另一方吞吐）与本场景相近；报告了 Thor 上 MPS 基线变差。
  - 不成立：两路任务共享同一 VLM 和同一观测，所以可以共享 prefill、合批 decode；本场景是两个独立模型（ImageWAM 与 27B VLM），无法共享 KV；只有均值频率，没有尾延迟。
- **精读笔记**：[notes/2026-oxygen.md](notes/2026-oxygen.md)。卡片与笔记不一致处以笔记为准。

### D28. FiS-VLA

**FiS-VLA** · 2025 · NeurIPS 2025（arXiv 页面无 comments；neurips.cc 2025 poster 页见于检索结果但未打开 †）· [arXiv 2506.01953](https://arxiv.org/abs/2506.01953)
Fast-in-Slow: A Dual-System Foundation Model Unifying Fast Manipulation within Slow Reasoning
Hao Chen et al.

- 场景：RTX 4090；System 1 复用 LLM 最后几层 transformer block，System 2 是完整 LLM（§3.2）；"the robot's hardware does not support deploying the two systems on separate GPUs for parallel inference"（§3.3）。
- 目标与指标：控制频率（Hz）、成功率。
- 核心观察：S2 : S1 = 1 : 4 最好（§4.2，扫 1:1–1:8，数值仅图 3）；动作块 1 时实测 21.9 Hz（CogACT 9.8、π0 13.8、OpenVLA 6.3，表 1）；动作块 8 时 117.7 Hz，附录 B.1 称之为 "theoretical control frequency"。
- challenge：单 GPU 上两系统不能并行。
- 机制：参数共享，让快系统成为慢系统的一部分；训练时异步采样降低 System 2 频率（§3.3）。单 GPU 上两者如何交错调度未写。
- 基线与评估：其他 VLA 的 Hz 与成功率。
- 与本场景的关系：成立的是"单设备承载双系统"的前提；不成立的是两系统共享参数、以固定比例交错，不存在两个独立模型的并发与干扰；117.7 Hz 是理论值。

### D29. RTC

**RTC** · 2025 · NeurIPS 2025（arXiv comments："Published in NeurIPS 2025"，未在论文集页面核对 †）· [arXiv 2506.07339](https://arxiv.org/abs/2506.07339)
Real-Time Execution of Action Chunking Flow Policies
Kevin Black et al.

- 场景：π0.5，50 Hz 控制（Δt = 20 ms），H = 50；推理在 RTX 4090（bf16，5 去噪步），LAN 远程推理加 10–20 ms，起始延迟 d ≈ 6（§4.2，附录 A.3）；注入 +100 / +200 ms 对应 d ≈ 11 / 16。
- 目标与指标：average task throughput（完成比例 / episode 时长，§4.2）；控制步与墙钟时间下的累计进度（图 6）。
- 核心观察：网络推理延迟 vanilla π0.5 76 ms、RTC 97 ms、BID（N = 16）115–223 ms（表 1，附录 A.3，10 次取平均，前 5 次 warm-up）；RTC 对注入延迟 "completely robust"，同步推理线性退化（§4.2，图 6）；比同步快 20%（图 1 说明）；temporal ensembling 在 +100 / +200 ms 下触发保护性停机。
- challenge：推理期间执行器必须继续出动作，新块要与已提交的动作衔接。
- 机制：执行器侧 deadline d = ⌊δ/Δt⌋，冻结前 d 步，对后续步做 inpainting 引导。
- 基线与评估：同步执行 s = 25、temporal ensembling、BID；10 次 × 任务 × 方法，共 480 episode。
- 与本场景的关系：成立的是"执行器在推理延迟内继续消耗已有动作"这一结构（本场景 16 步、每 200 ms 消耗 4 步）；不成立的是推理放在远端独占 GPU，没有共置；它把延迟当作给定输入，不讨论如何让延迟变小或稳定。

### D30. SmolVLA

**SmolVLA** · 2025 · 预印本 · [arXiv 2506.01844](https://arxiv.org/abs/2506.01844)
SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics
Mustafa Shukor et al. †（作者名未在本次打开的页面核对）

- 场景：SO100 机械臂，30 fps（Δt = 33 ms）；RobotClient / PolicyServer 拆分，策略 "can be run on a remote server, possibly with GPUs"（图 2 说明）；推理硬件未写。
- 目标与指标：成功率、任务完成时间、固定时间内搬动方块数（§4.6，图 5）。
- 核心观察：同步 / 异步完成时间 13.75 ± 2.42 / 9.70 ± 2.95 s，"around 30%" 更快；固定时间内 9 / 19 块；成功率 78.3 / 73.3（平均，图 5a）。没有测延迟。
- challenge：同步推理时机器人空等。
- 机制：队列剩余比例低于阈值 g 时发新观测；避免空等的条件 g ≥ (E[ℓ_S]/Δt)/n（§3.3）；延迟模型 E[ℓ] ≃ E[ℓ_S]，忽略通信。
- 基线与评估：同步推理。
- 与本场景的关系：g 的条件与本场景"16 步块、4 步/200 ms"的预取余量是同一形式；不成立的是单模型、远端、没有第二个模型共置，也没有尾延迟。
