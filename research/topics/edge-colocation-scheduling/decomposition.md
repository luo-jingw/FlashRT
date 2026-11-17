# edge-colocation-scheduling：30 篇论文的论证拆解

- 来源：[papers.md](papers.md) 的 A01–D30 卡片；5 篇精读笔记（XSched、Hummingbird、MuxWise/Drift、Serafini 第 6 章、OxyGen）与卡片不一致时以笔记为准；"洞察"与"贡献"取自论文摘要与引言原文。
- 原文版本：REEF、Orion、Clockwork、Paella、XSched、LithOS、BLESS 用会议版 PDF；GCAPS 用 ECRTS 2024 LIPIcs 版；Hummingbird arXiv v2；DistServe、Sarathi-Serve 用 OSDI '24 版；semi-PD、Nexus、Bullet（arXiv 当前标题 "Boosting LLM Serving through Spatial-Temporal GPU Resource Sharing"）用 arXiv 当前版；MuxWise arXiv v3；DuetServe 用 PMLR v306 PDF；Miriam arXiv v1；RT-mDL、BWLOCK++、TCAD 用会议/期刊版；Twill、Edge GPU 隔离对比、NavGPT-3、FiS-VLA、RTC、SmolVLA、OxyGen 用 arXiv 版（OxyGen 为 v5）；D24 用 PR 正文；D25 只有 neurips.cc 摘要。
- "本场景"指 [survey.md](survey.md) §0.2：Jetson AGX Thor（20 SM，LPDDR5X 约 273 GB/s 统一内存），执行器 ImageWAM 37.5 ms、截止期 200 ms，规划器 Qwen3.8-27B，prefill 每块 2048 token 约 1.17 s，decode 约 27 ms/token；可用控制为 MPS 百分比、green context、MPS 客户端优先级、stream 优先级。
- 引文保持英文原文并注明位置；"判断"是本综述的判断，不是论文结论。

## A. GPU 共享与抢占

### A01 REEF

- 问题：GPU 上同时服务 real-time（RT）与 best-effort（BE）DNN 推理时，RT 要"first-class citizens"、系统又要 work conserving，商用 GPU 没有高效抢占。
- 观察：多 stream 并发时 RT 时延从独占约 4 ms 升到接近 50 ms（§1，Fig.1b，MI50，VGG RT + ResNet BE）；DNN 推理 kernel 大多幂等（§2，Table 1）。
- 洞察："A key insight of REEF is that each kernel in DNN inference is mostly idempotent."（§1）
- 贡献：(1) 刻画 DNN 推理 kernel 的幂等性等特征与现有调度的问题；(2) reset-based preemption，与被抢占 kernel 数无关；(3) dynamic kernel padding；(4) AMD 与 NVIDIA 上的实现与评估（§1 Contributions）。
- 机制：直接杀掉并清空 BE kernel、之后重跑，不保存上下文；再把 BE kernel 填进 RT kernel 剩下的 CU。
- 证据：DISB A 下 RT 时延开销 0.5%、吞吐 ×1.60，而 SEQ / GPUStreams RT 时延 ×1.95 / ×1.84（§7.2，Fig.11）；NVIDIA V100 上 REEF-N 抢占延迟比 wait-based 快至多 12.3×（§7.5）。
- 对本场景：NVIDIA 版 REEF-N 不能杀掉正在运行的 kernel，只能清空 host 队列，而本场景规划器 prefill GEMM kernel 远长于 REEF 中 10–255 µs 的 kernel。

### A02 Orion

- 问题：多个 DNN 作业共享一块 GPU 时，保持一个高优先级（HP）作业接近独占的尾时延，同时用剩余资源跑 best-effort（BE）作业。
- 观察：MobileNetV2 训练的计算吞吐与带宽利用率均值分别低于 40% 和 55%（§1，Fig.1）；V100 上 Conv2d 与 BN2d 并发比顺序快 1.41×，两个 Conv2d 并发 0.98×（§3，Table 2）。
- 洞察："Despite the prevalence of GPU sharing techniques, current approaches are not sufficiently fine-grained or interference-aware to maximize GPU utilization while minimizing interference at the granularity of 10s of 𝜇s."（Abstract）
- 贡献：论文未列贡献（§1 以叙述给出系统与结果）。
- 机制：拦截 kernel launch，按算子粒度调度；BE kernel 只在 SM 需求小于阈值且 compute/memory 特征与当前 HP kernel 相反时提交，并限制在途 BE kernel 预计总时长（§5.1）。
- 证据：推理 + 训练共置时 HP p99 在理想值 14% 以内，比 REEF 低 2.33×；REEF 在 Apollo / Poisson 下为理想的 3.44× / 2.5×（§6，Fig.6–7）。
- 对本场景：compute/memory 二分标签对整体带宽密集的 LLM decode 不够细（Hummingbird §3 测得 Orion 下 Llama-8B TPOT 波动为独占的 10.9×），而本场景按 SM 划分后的剩余干扰正是 decode 带宽。

### A03 Clockwork

- 问题：模型服务系统要稳定满足低时延 SLO，但现有架构的反应式技术压不住由执行时间不可预测带来的尾时延。
- 观察：V100、batch 1 时 DNN 推理 p99.99 与中位数相差 0.03% 以内（§2，Fig.2a）；线程池并发吞吐最多 +25%，尾时延上升约 100×（§2，Fig.2b）。
- 洞察："Yet the underlying execution times are not fundamentally unpredictable—on the contrary we observe that inference using Deep Neural Network (DNN) models has deterministic performance."（Abstract）
- 贡献：(1) 证明可预测性是 DNN 推理的基本特征；(2) "consolidating choice" 设计方法；(3) Clockwork 分布式服务系统的设计与实现；(4) 实验表明每 GPU 支持数千模型并显著降低尾时延（§1 main contributions）。
- 机制：每块 GPU 一次只跑一个推理，所有并发、排队、加载决策集中到上层控制器，按预测执行时间提前排程，预计超 SLO 的请求提前拒绝。
- 证据：100 ms 目标下 99.9999% 请求达标（Abstract）；SLO 收紧时 Clipper / INFaaS goodput 崩溃而 Clockwork 维持（§6.1，Fig.5）。
- 对本场景：Clockwork 以放弃 GPU 内并发换可预测性，而本场景只有一块 GPU，执行器若要等规划器一整块约 1.17 s 的 prefill forward 就会超过 200 ms 截止期。

### A04 Paella

- 问题：模型服务框架停留在高抽象层、依赖黑盒 GPU 硬件调度，在 GPU 被多作业共享时既低效又有框架开销。
- 观察：一次性提交全部 kernel 会占满 32 个硬件队列造成 HoL 阻塞，合成负载下最坏只用到 32/176 = 18% 的并发度（§2，Fig.2）；Triton 框架开销对某些模型与执行时间相当（Fig.3）。
- 洞察："In this paper, we argue for co-designing the model compiler, local clients, and the scheduler to bypass the built-in GPU scheduler and enable software control of kernel execution order."（Abstract）
- 贡献：(1) 指出现有服务框架受周边组件与接口拖累；(2) 编译器-服务协同设计，插桩 kernel 以获得对 GPU 调度的可见性与控制；(3) 把调度从 GPU 抽离的服务架构，覆盖接收、下发、返回全链路（§1 contributions）。
- 机制：TVM 编译期插桩让 kernel 开始/结束时通知 host，host 单核 dispatcher 只在资源可用时逐个下发 kernel，策略为 SRPT 加公平阈值。
- 证据：相对 Triton 可持续负载至多 58×、时延低 11×；仅调度算法比基线多 1.4× 负载、时延低 1.35×（§1、§7.1，Fig.11）。
- 对本场景：所有模型须经 TVM 编译插桩，本场景规划器 kernel 来自 TensorRT / 厂商库 / 手写融合 kernel，插桩路径不直接可用。

### A05 XSched

- 问题：GPU、NPU、ASIC、FPGA 等 XPU 缺乏灵活调度能力，多任务环境下无法满足优先级、公平等需求，现有方案不可移植。
- 观察：原生硬件调度下单个后台任务让前台 P99 变成单独运行的 1.60–2.19×（§7.2，Fig.9）；Jetson Orin 上即使 GPU 与 DLA 计算独立，只调度 DLA 时前台 P99 仍为 1.67×（§7.2，Fig.10）。
- 洞察："We propose establishing a unified abstraction with a multi-level hardware model for preemptive XPU scheduling, which effectively conceals the differences and complexities in hardware capabilities and software stacks, ensuring generalizability."（§1 Key idea）
- 贡献：论文未列贡献（§1 以 Key idea / Our approach 叙述：XQueue 抽象、三级硬件模型、XSched 框架、NVIDIA / Intel 上的 Lv2/Lv3 新技术、10 种 XPU 上的实现与评估）。
- 机制：可抢占命令队列 XQueue；Lv1 progressive launching 只保留不超过阈值的在途命令，Lv2 / Lv3 分别撤回已下发命令、中断运行中命令。
- 证据：固定优先级下前台 P99 相对原生最多降 2.10×（Abstract、§1；§7.2 写 2.11×）；跨 XPU 统一优先级使 Orin 前台 P99 从 1.67× 降到 1.18×（§7.2，Fig.10，平台对应关系见笔记）。
- 对本场景：论文 §9 指出 CUDA graph 是单命令任务、Lv1/Lv2 对其失效，而本场景规划器一块 prefill forward 约 1.17 s。

### A06 LithOS

- 问题：数据中心 GPU 利用率低，需要对应用透明、细粒度地管理 GPU 资源，同时兼顾利用率、能效与隔离。
- 观察：Meta 生产集群计算利用率平均 27%（§3.1，Fig.1，arXiv v1 文本）；训练 kernel 常持续数毫秒，DLRM 超过 30 ms（§5，Fig.11）。
- 洞察："Transparent, fine-grained GPU resource management that maximizes utilization, energy efficiency, and isolation requires an OS approach."（Abstract）
- 贡献：(1) Meta 推理服务研究；(2) TPC Scheduler 与 TPC Stealing；(3) 透明的 Kernel Atomizer；(4) 动态 right-sizing；(5) 按 kernel 的透明功耗管理；(6) LithOS 设计；(7) 跨 ML 环境评估（§1 contributions）。
- 机制：拦截 CUDA Driver API，按 TPC 给每个 kernel 分配计算单元，并把 kernel 的 thread block 切成 atom 分批下发以缩短 HoL 阻塞。
- 证据：推理叠加尾时延比 MPS 低 13×、比最优 SotA 低 4×（Abstract）；推理 + 训练时 HP 在理想 20% 以内，而 MPS 5.83×、TGS 1.41×、REEF 8.93×（§8.1，Fig.17）。
- 对本场景：只在 A100（独立 HBM）上评估，且不控制内存带宽，而本场景按 SM 划分后的剩余干扰来自统一内存上的 decode 带宽。

### A07 GCAPS

- 问题：带 GPU 段的实时任务需要可分析的抢占式优先级调度，而现有同步协议方法把 GPU 段当作不可抢占，导致长阻塞与优先级反转。
- 观察：Tegra 默认驱动对各进程 TSG 做轮转时间片、不看优先级（§2）；TSG 切换开销平均超过 200 µs，runlist 更新峰值约 1 ms（Fig.12–13，数字来自卡片所读 arXiv v1 HTML）。
- 洞察："Our approach exerts control over GPU context scheduling at the device driver level and enables preemption of GPU execution based on task priorities by simply adding one-line macros to GPU segment boundaries."（Abstract）
- 贡献：(1) GPU context 感知的抢占式优先级调度，并可为 GPU 段单独赋优先级；(2) 考虑自挂起 / 忙等与上下文切换开销的响应时间分析；(3) 对默认 Tegra 驱动轮转策略的响应时间分析；实现于最新 Tegra 驱动并开源（§1 contributions）。
- 机制：GPU 段前后插宏经 ioctl 通知驱动，修改驱动按优先级从 runlist 移除低优先级 TSG。
- 证据：可调度率比已有方法高至多 40%（Abstract，Fig.8）；Xavier 案例任务 1 的 MORT / WCRT 从 26.13 / 60 ms（tsg_rr_busy）降到 9.68 / 16 ms（Table 5）。
- 对本场景：按 GPU 段整体时间切分意味着两个模型不同时在 GPU 上运行，且需改驱动，Thor 驱动兼容性论文未涉及。

### A08 BLESS

- 问题：多个轻负载应用按配额共享一块 GPU 时，现有时间或空间共享无法高效、准确地兑现配额。
- 观察：A100 上 VGG11（1/3 配额）+ ResNet50（2/3 配额）的某请求时延，时间共享 17.1 ms、空间共享 11.5 ms、消除气泡后 10.1 ms（§1，Fig.1）。
- 洞察："We observe that the performance of the multi-user system is often underestimated because of the existence of unused GPU “bubbles” and can be enhanced by squeezing the bubbles."（Abstract）
- 贡献：(1) 多应用共享 GPU 时"气泡"的分析；(2) 时空共享下 kernel 执行的精确建模；(3) 细粒度调度加精确资源管理的系统方案（§1 three-fold contributions）。
- 机制：每个客户端预建多个不同 SM 数的 MPS context，运行时把来自不同应用的 kernel 组成 kernel squad，按离线估计器选出时空共享配置并切换 context 下发。
- 证据：平均时延比 SOTA 降 21.1–37.3%；两个 BERT 共置降 39.1%，四个降 41.2%（§1、§6）。
- 对本场景：目标是按配额公平，而非一方有硬截止期，且应用都是小推理模型、A100 独立显卡。

### A09 Hummingbird

- 问题：在一块 NVIDIA GPU 上共置高优先级 LLM 服务与低优先级作业，如何在保证 SLO 的前提下最大化利用率。
- 观察：A100 上 Llama-8B 的 kernel 从 5 µs 到 7.49 ms，但 99.999% 的 thread block 在 390 µs 内完成（§1，Fig.3）；Orion 下 Llama-8B TPOT 波动为独占的 10.9×（§3，Fig.5）。
- 洞察："The core design principle of SLO-oriented scheduling is to ensure that high-priority tasks execute with strict performance isolation, while low-priority tasks opportunistically harvest idle time slices."（§1 Major insights）
- 贡献：论文未列贡献（§1 以 Major insights 叙述）。
- 机制：把低优先级 kernel 按一波 thread block 切成 split-kernel（平均 77 µs），高优先级到达时停止下发后续分段，只在高优先级的小气泡中插入低优先级分段。
- 证据：单 GPU 上 SLO 达成率平均比 LithOS 高 5.6×、比 REEF 高 3.0×，低优先级吞吐为 REEF 的 1.9×，HP 变慢 <1%（§6.1，Fig.10）；摘要中的 9.7× 来自多 GPU 对 REEF（§6.3）。
- 对本场景：高低优先级角色相反——论文保护的是 LLM 服务、牺牲离线批处理，而本场景保护小执行器、规划器 LLM 本身也有 5.33 s 周期要求。

## B. LLM 服务的阶段调度

### B10 DistServe

- 问题：LLM 服务在同一 GPU 上合批 prefill 与 decode，在严格 TTFT / TPOT 要求下只能牺牲其一或过度配置 GPU。
- 观察：单 A100、13B 模型下合置系统 90% attainment 的 goodput 约 1.6 rps，prefill / decode 分开各占一卡时分别为 5.6 / 10 rps，2P1D 折合约 2.1×（§1，Fig.1）。
- 洞察："We find that this strategy not only leads to strong prefill-decoding interferences but also couples the resource allocation and parallelism plans for both phases."（Abstract）
- 贡献：(1) 指出 prefill-decode 干扰与资源耦合问题并提出分离两阶段；(2) 自动选择 goodput 最优 prefill / decode 部署方案的放置算法；(3) 真实负载上的全面评估（§1 contributions）。
- 机制：prefill 与 decode 实例放到不同 GPU，用模拟器加搜索为每阶段选并行配置与副本数，并按带宽放置降低 KV 传输开销。
- 证据：相对 vLLM、DeepSpeed-MII 最多 7.4× 请求率或 12.6× 更紧 SLO，同时 >90% 请求满足时延约束（Abstract、§1）。
- 对本场景：隔离靠多 GPU 物理分离，而 Thor 只有一块 20 SM 的 GPU。

### B11 Sarathi-Serve

- 问题：LLM 服务中 prefill 与 decode 迭代交错，难以同时获得高吞吐与低 TBT 尾时延。
- 观察：prefill 优先调度（vLLM、Orca）产生持续数秒的 generation stall（§1，Fig.1）；不切块混合 batch 让 TBT 最多升 28.3×（§4.2，Fig.9）。
- 洞察："Takeaway-2: Decode batches operate in memory-bound regime leaving compute underutilized. This implies that more tokens can be processed along with a decode batch without significantly increasing its latency."（§3.1）
- 贡献：(1) 指出现有 LLM 服务系统在吞吐-时延权衡上的问题；(2) chunked-prefills 与 stall-free batching 两项技术；(3) 跨模型、硬件、并行策略的评估（§1 main contributions）。
- 机制：每轮先放全部 decode，再在 profiling 得到的 token budget 内放入 prefill 块，使每次迭代时长有界。
- 证据：strict SLO 下 capacity 比 Orca 高至多 4.0×、比 vLLM 高 3.7×（Yi-34B）；PP 下 6.3× / 4.3×（LLaMA2-70B）（§5.1）。
- 对本场景：token budget 由同一 batch 的 TBT 推导，而本场景受保护的是另一个模型的截止期，且规划器块切得更小会改变输出。

### B12 semi-PD

- 问题：分离式 LLM 服务解决了干扰，但带来权重复制、KV 传输、存储失衡与难以调整资源等存储问题。
- 观察：分离部署中 prefill 实例最多只用 25% KV 存储，最多 89.33% 显存浪费（§3.1，Fig.3）；存储耗尽使 P90 TPOT 升 3×（§3.1，Fig.4）。
- 洞察："The key insight is that we should adopt the disaggregated computation feature of the disaggregated system and reserve the unified storage feature of the unified system."（§1）
- 贡献：(1) 分析统一与分离设计并提出阶段级计算分离 + 统一存储的设计与原型；(2) 低开销 worker 切换机制与 SLO 感知切换算法；(3) 实现与评估（§1 contributions）。
- 机制：prefill 与 decode 各为一个进程，用 MPS 百分比划分 SM，常驻进程持有权重与 KV 经 IPC 共享，SLO 感知算法按窗口调整 SM 比例。
- 证据：DeepSeek 上平均端到端时延降 1.27–2.58×（§7.2.1，Fig.14）；Llama 上 >90% attainment 时可承受请求率比 vLLM-S / vLLM-D / DistServe 高 1.55× / 1.72× / 1.62×（§7.2.2，Fig.15）。
- 对本场景：不分析两进程之间的带宽争用，且两进程同属一个模型、SLO 对称、调度为 FCFS 无优先级。

### B13 MuxWise（Drift）

- 问题：LLM 服务须同时满足 prefill 与 decode 的 SLO；分离部署导致资源闲置或计算冗余，chunked prefill 在 SLO 合规与利用率之间两难。
- 观察：8×A100、Llama3-70B 上 chunked prefill 延迟到 4K budget 才开始线性增长，此时 505 ms，而 100 ms TBT 对应约 256 budget（v3 §1、§2.3，Fig.6a）；空间复用下 decode 争用 slowdown 约 0–30%（v3 §3.3.1，Fig.11）。
- 洞察："To address these issues, an efficient serving system should dynamically adapt compute allocation, decouple compute from memory management, and execute prefill and decode independently."（v3 Abstract）
- 贡献：(1) 分析已有工作，提出高 goodput LLM 服务的关键需求；(2) 提出 PD multiplexing 范式及其设计；(3) 多样负载下的评估（v3 §1 contributions）。
- 机制：同一进程内用 green context 把 SM 分给 prefill 与 decode，decode 拿 solo-run 预测 × contention guard 下刚好满足 TBT 的最少 SM，prefill 按层下发以消除气泡并允许在层边界重分区或抢占。
- 证据：SLO 保证下峰值吞吐平均 2.20×、最多 3.06×（v3 Abstract）；70B goodput 对 chunked / LoongServe / SGLang-PD 为 3.06× / 2.62× / 1.62×（v3 §4.2.3，Fig.15）。
- 对本场景：争用上界（A100 20%、H100 30%）是在 108 SM 的 HBM GPU 上测得，而本场景 20 SM、273 GB/s 统一内存上 decode 带宽是主要剩余干扰，且受保护方是同一模型的 decode。

### B14 Nexus

- 问题：引擎级 PD 分离需要更多 GPU，chunked prefill 又引入阶段干扰；能否在单个服务引擎、单块 GPU 内实现分离。
- 观察：prefill SM 从 30% 到 40% 延迟降 >25%，70% 到 80% 只降约 10%（§3.2，Fig.5）；prefill KV 长度从 2000 增到 10000，同一 decode batch 延迟升 36%（§3.3，Fig.6）。
- 洞察："Second, we find that GPU resource exhibits diminishing returns—beyond a saturation point, increasing GPU allocation yields negligible latency improvements. This insight enables us to split a single GPU’s resources and dynamically allocate them to prefill and decode on the fly, effectively disaggregating the two phases within the same GPU."（Abstract）
- 贡献：(1) 指出现有方案局限并提出引擎内 PD 分离；(2) 轻量自适应调度，支持亚秒级重分区与阶段感知优先级；(3) 作为 vLLM 扩展实现并评估（§1 contributions）。
- 机制：启动时预建所有 green context 分区，用含带宽份额项的 max(计算, 访存) 成本模型做两阶段贪心搜索选 SM 比例，并加迟滞阈值防振荡。
- 证据：对 vLLM 吞吐最多 2.2×、TTFT 低 20×、TBT 低 2.5×；对 SGLang 2× / 2× / 1.7×；以一半 GPU 超过 vLLM-P/D 1.4× 吞吐（Abstract、§6.2）。
- 对本场景：受害方是同一模型的 decode、施压方是 prefill，L20 为 GDDR6 独立显卡，与本场景执行器受规划器（含 decode）带宽挤压的方向和硬件都不同。

### B15 Bullet

- 问题：LLM 服务中计算密集的 prefill 与访存密集的 decode 不匹配，混合 batch 在吞吐与时延间做低效取舍，GPU 资源大量闲置。
- 观察：1024 token 时 wave quantization 让整层理论 SM 空闲 19.4%、O-proj 40.7%（§2.2.1，Table 1）；整层实测计算利用率只有 70–76%（§2.2.2，Fig.2）。
- 洞察："we identify two key root causes: 1) the prefill phase suffers from suboptimal compute utilization due to wave quantization and attention bottlenecks, and 2) hybrid batching disproportionately prioritizes latency over throughput, wasting both compute resources and memory bandwidth."（Abstract）
- 贡献：(1) 识别并系统分析利用率低下的原因与机会；(2) 时空共享阶段的精确建模与面向时延的细粒度资源控制；(3) 集成进现有框架的服务系统；(4) 真实负载评估（§1 contributions）。
- 机制：MPS 下用 libsmctrl 给每个引擎的 stream 设 SM mask（重配置平均 4.1 µs），用含争用项的性能模型按 P90 TTFT / TPOT 动态重分 SM。
- 证据：相对 vLLM、SGLang、Nanoflow 平均吞吐 1.26×、最多 1.55×（Abstract）；单 GPU 对 SGLang-1024 吞吐 1.09×（§4.2.1）。
- 对本场景：libsmctrl 依赖未公开驱动接口，Thor 上可用性未知，且作者自述低算力 GPU 上 decode 需要更大 SM 份额、增益缩小（§4.6）。

### B16 DuetServe

- 问题：LLM 服务合置两阶段会因 prefill 干扰恶化 TBT，跨 GPU 分离又浪费资源（模型复制、KV 传输）。
- 观察：H100 上 8192 token 的纯 prefill batch "consistently exceed 180 ms"（§3，Fig.1b）；约 20% 的 SM 即可达到约 60% 峰值 HBM 带宽（§3，Fig.3a）。
- 洞察："Its key idea is to decouple prefill and decode execution only when needed through fine-grained, adaptive SM partitioning that provides phase isolation only when contention threatens latency service level objectives."（Abstract）
- 贡献：(1) attention-aware roofline 模型预测迭代时延；(2) 在 TBT 约束下选最优 SM 划分的分区优化器；(3) 消除 CPU-GPU 同步开销的无中断执行引擎（§1 contributions，三个组件）。
- 机制：默认合置运行，预测到 TBT 超界时才用 libsmctrl 按 TPC 粒度给 decode 刚好够的 SM，并连续发 k 步 decode CUDA Graph 避免每步同步。
- 证据：总吞吐最多 1.3×（Abstract）；Azure-Code QPS 16 时 13.57 vs 12.43 req/s，TBT <150 ms，而 vLLM 与 SGLang-Chunked >200 ms（§5）。
- 对本场景：单进程两 stream 共享一个模型，而本场景是两进程两模型，且论文未分析并发 prefill 与 decode 之间的 HBM 争用。

## C. 边缘 SoC 上的多模型实时推理与内存带宽争用

### C17 Miriam

- 问题：边缘 GPU 上多个不同关键性的 DNN 并发推理，关键任务要优先、整体吞吐要高，而边缘 GPU 缺硬件级资源管理（引言称 MPS / MIG 不可用）。
- 观察：Tango benchmark 中只有 7.4% 的 kernel 可直接改 grid/block 而不改计算调度（elastic kernel 一节）；Multi-stream 下关键任务时延为 Sequential 的 1.95×（2060）/ 2.02×（Xavier）（Fig.8）。
- 洞察："Our design is based on the key observation that the latency degradation of co-running DNN kernels is mainly caused by two dominant factors, namely intra-multi-processor (SM) resource contention and inter-multi-processor resource contention."（§1，arXiv v1）
- 贡献：论文未列贡献（Fig.3 图注以 "Green boxes are our contributions" 标出 elastic-kernel generator 与 runtime coordinator）。
- 机制：离线把普通任务 kernel 经 source-to-source 变换成 grid/block 可调的 elastic kernel，在线只用关键 kernel 剩下的 SM 资源填充其分片。
- 证据：相对 Sequential 吞吐提升至多 92%、关键任务时延开销 <10%（Abstract）；MDTB A 上关键任务时延仅增 21% / 28%，而 Multi-stream 为 1.95× / 2.02×（Fig.8）。
- 对本场景：普通任务以吞吐为目标且不控制内存带宽，而本场景规划器有自己的时延要求、剩余干扰来自 decode 带宽。

### C18 RT-mDL

- 问题：资源受限的边缘平台要并发执行多个有不同实时 / 精度要求的 DL 任务，现有 DL 框架缺少实时支持。
- 观察：所有优先级分配下最差任务错过率仍为 99.33%（Xavier）、72.40%（TX2）（§3，Fig.3）；VGG11 精度降 1% 延迟降 27.65%（Fig.2，Xavier）。
- 洞察："To tackle the challenge of resource contention, our key idea is to find a solution that jointly optimizes the scaling levels of all DL tasks as well as the task priorities."（§3 末）
- 贡献：论文未列贡献（§1 以叙述给出：存储受限的模型缩放、模型变体与优先级联合优化、独立 CPU/GPU 队列与 GPU packing 的调度器）。
- 机制：生成多级模型变体，用 MOEA 搜索"变体 + 优先级"组合，错过率由 Random Forest 代理模型预测；任务拆成 CPU / GPU 子任务分别排队。
- 证据：相对规则式调度基线错过率最多降 40.12%，精度损失 1.7%（Abstract）；轻负载下 AlexNet-2 错过率 6.15%，SPO-NMC 8.94%、UM-NMC 16.29%（Fig.17a）。
- 对本场景：主要控制量是降模型精度换延迟，而本场景规划器输出不允许被改变。

### C19 Twill

- 问题：compound AI 系统在移动边缘平台上需并发运行 DNN、transformer 与 LLM，请求按未知顺序动态到达。
- 观察：transformer 算子大多不被 DLA 支持，只能在 GPU 上跑（Fig.5，DLA 兼容性实测）；DNN 与 transformer 同在 GPU 上时两者延迟都升高（Fig.3，仅图）。
- 洞察："Orchestrating cAI workloads requires a run-time adaptive scheduling strategy that considers dynamic inference requests and migrates diverse inference tasks among GPU and DLA clusters to minimize overall inference latency."（§I）
- 贡献：(1) Twill 运行时框架；(2) 判定任务对 GPU / DLA 亲和度的在线启发式模型；(3) 任务到 cluster 的映射与迁移、按优先级冻结任务、DVFS 功耗封顶；(4) Jetson Orin NX 上评估（§I contributions）。
- 机制：在线检查每层与 DLA 的兼容性，按亲和度映射与迁移任务，每个 cluster 同时只跑一个模型，必要时冻结低优先级任务并调 DVFS。
- 证据：相对 Tango、Mapformer、Band 等方法平均推理延迟降低 54%，同时满足功耗预算（Abstract）。
- 对本场景：靠"一个 cluster 只跑一个模型"回避 GPU 内并发，而本场景两个模型都只能用 GPU，争用无法靠换加速器回避。

### C20 Edge GPU 隔离对比

- 问题：安全关键的多模型集成推理需要可预测的推理时间，现代 NVIDIA GPU 的 MPS、MIG、Green Context 能提供多少时间隔离。
- 观察：Orin Nano 上 MPS 的最大稳定频率与独占相同（ConvNeXt Large 33 / 33），GC 4 SM 分区降为 18（Table III）；Orin AGX 固定 1.02 GHz 后 GC 超时比例大幅下降（Fig.10）。
- 洞察："It is observed that MIG provides a high level of isolation. At the same time, Green Contexts represent a promising alternative for edge devices by enabling fine-grained SM allocation with low overhead, albeit without memory isolation."（Abstract）
- 贡献：论文未列贡献（测量对比，不提出新机制）。
- 机制：一个进程以最大稳定频率运行、另一进程逐步加频，统计前者超时比例，比较 MPS、MIG（A100）与 GC（Jetson）。
- 证据：Orin 上并发受功耗限制而降频，使 GC 隔离变差；固定频率后 GC 接近 A100 上 MIG 的隔离水平（Fig.6、Fig.9–10）；GC 只隔离 SM、不隔离内存（结论节）。
- 对本场景：两个进程跑同一小视觉模型、角色对称，没有 LLM 长 prefill，也没有评估 MPS 客户端优先级。

### C21 BWLOCK++

- 问题：集成 CPU-GPU SoC 上，CPU 内存密集任务通过共享主存拖慢实时 GPU kernel。
- 观察：Jetson TX2 上 histo 在 CPU 共跑者干扰下执行时间增至 3.3×（§1，Fig.1）；摘要称最坏约 3×。
- 洞察："As a result, even though the platform offers plenty of raw performance, no real-time execution guarantees can be provided if the system is left unmanaged."（§1）
- 贡献：(1) 用带宽限流保护 GPU 实时任务免受 CPU 内存密集任务干扰；(2) 识别限流与 CFS 的负反馈并提出 TFS；(3) 自动 GPU kernel 插桩；(4) TX2 上实现与评估；(5) 接入 CPU 侧可调度性分析（§1 contributions）。
- 机制：拦截 CUDA API，在 GPU kernel 执行期间持有内存带宽锁，期间由 PMC 计 LLC miss 的 per-core 调节器限流 best-effort CPU 核。
- 证据：GPU benchmark 在共跑下接近 solo 性能（Fig.1）；TFS 相对 CFS 系统 throttle 时间减少 39%（TFS-1）/ 62%（TFS-3）（Fig.10），CPU 吞吐损失最多减 75%（§1）。
- 对本场景：只处理 CPU→GPU 方向且假设只有实时任务用 GPU，本场景规划器对执行器的 GPU 内带宽争用不在其模型内。

### C22 动态带宽分配

- 问题：静态带宽限流因 WCET 估计悲观，过度压制 best-effort CPU 任务的带宽。
- 观察：要把 histo 的 WCET 膨胀限制在 10%，须把 BE 核带宽降到最大吞吐的 13%（§I，引 §VII 评估）。
- 洞察："it is important to notice that the contention caused by BE applications at run-time can sometimes be significantly less than the worst-case measurable one."（§I）
- 贡献：(1) 按 thread block 聚类估计 GPU kernel 运行进度；(2) 基于剩余线程组与带宽阈值的测量式 WCET 估计；(3) 在线重算 BE 带宽并保证 kernel 在原 WCET 内完成；(4) Jetson TX2 上实现验证（§I key contributions）。
- 机制：运行时读取已完成 block，按剩余工作重算本周期可给 CPU 的带宽（FAIR / GREEDY / SMOOTH），通过 BWLOCK++ 内核模块执行。
- 证据：相对静态预算 NOMINAL，best-effort 内存请求数提升 74%–195%（libquantum）、21%–42%（gcc）、0%–13%（sjeng）（Table III）。
- 对本场景：假设 GPU 一次只跑一个不可抢占 kernel、尽力而为方是 CPU 任务，而本场景尽力而为方是 GPU 上的规划器，没有对 GPU 进程按带宽限流的接口。

### C23 Serafini 博士论文第 6 章

- 问题：NVIDIA 异构 SoC 上 GPU 与 CPU 共享主存，GPU 引起的内存干扰威胁 CPU 关键任务的可预测执行。
- 观察：Thor 上合并写流量使 CPU slowdown 2.5×（Fig.6.10b）；512 B 跨步写使 CPU 慢约 5×，且 1 个 SM 就能压满内存控制器（§6.4.4，Fig.6.12）。
- 洞察："our MBMS operates in the spatial domain: it dynamically limits the number of streaming multiprocessors (SMs) available to GPU kernels based on CPU workload criticality."（§1.2.3）；同节："This finding highlights the limits of spatial domain regulation and demonstrates that platform-specific phenomena require adaptive mitigation strategies."
- 贡献：学位论文 §1.2 列三项贡献，对应第 6 章的是 §1.2.3：基于 Green Contexts 的空间域 MBMS，在 AGX Orin 与 AGX Thor 上评估，并发现 Thor 上单 SM 跨步写导致最高 5× CPU slowdown 的病态情形。
- 机制：拦截 cudaLaunchKernel，按 CPU 任务声明的 QoS 选择 Green Context 分区，把 GPU kernel 放到更少的 SM 上以降低其访存并行度。
- 证据：相对不调节，合并写 slowdown 从 2.5× 降到 1.6×（Thor）、1.55× 降到 1.2×（Orin）；Thor 跨步读 3× 降到 1.4×，跨步写不受调节（§6.4.4，Fig.6.9–6.12）。
- 对本场景：受保护方是 CPU 任务，而本场景受保护方是 GPU 上的执行器、干扰方主要是同一 GPU 上的规划器。

## D. 机器人双系统的部署

### D24 TRT-Edge-LLM PR #193

- 问题：在 Jetson Thor 上部署 InternVLA-N1-DualVLN，让 Qwen2.5-VL-7B 规划器（System 2）与轨迹专家（System 1）以不同频率重叠运行。
- 观察：System 2 空闲时 System 1 每条轨迹 48.1 ms；同进程 + 优先级 stream 时 55.9 ms；System 2 在另一进程时 117–122 ms（Validation，Latency and control rate 节）。
- 洞察："One process is what makes the priority stream effective — CUDA orders streams within a context, so across processes the GPU merely time-slices"（Validation）；同节："the trajectory head costs ~52 ms whichever planner shares the GPU, and what changes is how much the planner crowds it."
- 贡献：论文未列贡献（PR 正文列出三个评审要点：直接读取发布的 InternVLA 配置无需重打包；System 1 以 vendored 方式重写；运行时放在 experimental_models/ 且 System 1 使用独立上下文内存池）。
- 机制：单进程内 System 1 跑在高优先级 CUDA stream，规划器线程异步运行，System 1 始终读最新规划并报告 staleness；不用 MPS，不分 SM。
- 证据：NVFP4 规划器下 control rate 18.0 Hz，PyTorch bf16 阻塞式参考实现 4.8 Hz；最坏 tick 78 ms 对 331 ms（Validation）；闭环 SR NVFP4 67.8% 对 PyTorch 69.8%，p = 0.572（199 episode，McNemar）。
- 对本场景：规划器 prefill 只有 1024 token、约 75 ms，比本场景 4–6k token、1.2–2.2 s 的 prefill 小一个数量级以上，且未测 MPS 客户端优先级与 SM 划分。

### D25 DualVLN on Thor

全文未获得（OpenReview 返回 captcha），以下仅依据 neurips.cc 页面上的摘要。

- 问题：机器人上的 VLN 模型必须装进板载计算机，并依据仍然新鲜的观测行动（Abstract 首句）。
- 观察：TensorRT + NVFP4 移植使中位延迟降约 6.8×（规划）、7.9×（look-down 规划）、3.9×（轨迹生成），计时边界各不相同（Abstract）。
- 洞察："In this small simulation study, where the scene is held while the models compute, inference speed alone determines neither observation freshness nor navigation quality."（Abstract）
- 贡献：论文未列贡献（仅读到摘要）。
- 机制：把 DualVLN（7B System 2 + diffusion System 1）不重训移植到单块 Jetson Thor，并比较 System 1 输出复用次数等四种运行时设置。
- 证据：每个 System 1 输出只用一次（而非最多 4 个动作）使当前观测的 p95 年龄从 2.91 s 降到 0.99 s，System 1 service rate 相近；另 12 条路径上原实现 / 移植 / 最终设置成功 20 / 23 / 22 次（各 48 次）（Abstract）。
- 对本场景：仿真在模型计算期间冻结场景，延迟不影响世界演化，也没有截止期。

### D26 NavGPT-3

- 问题：如何把擅长推理的语言模型与提供稠密低时延控制的动作策略连接起来，并让机器人能对突发事件即时响应。
- 观察：语言模型自己导航时每 3–19 s 才做一次决策，交给 VLA 执行后最小反应时间降到每步 0.5–1 s（1–2 Hz）（Abstract、§1，Table 3）。
- 洞察："These results show that designing this embodied interface is central to connecting frontier language-model intelligence with low-level physical control."（Abstract）；§1："The world also does not pause while a model thinks"。
- 贡献：论文未列贡献（§1 以叙述给出 harness、OS-like runtime、NavGPT VLA 与评测结果）。
- 机制：推理、执行、监视各为一个有独立上下文与权限的线程，runtime 按优先级规则决定哪个线程拥有 motion authority，监视器可中断运动。
- 证据：真机 halt latency p95：规划器 + 工具 2150 ms，重叠 / 优先级 180 ms（表 10B）；RxR-CE 上 90.43 SR，对比人类 90.4（Abstract）。
- 对本场景：规划器是云端托管服务，与 VLA 不共用计算资源，"重叠"不产生 GPU 或带宽干扰。

### D27 OxyGen

- 问题：MoT 结构的 VLA 在设备上同时服务每帧动作与跨帧语言等多任务时，现有推理系统产生重复计算与资源争用。
- 观察：4090 上只做 KV 共享帧延迟 200.3 → 145.1 ms（N = 5，§4.3）；MPS 多进程并行在 Thor 上比顺序执行还慢（§4.2 观察 iii，具体降幅仅图）。
- 洞察："We identify isolated KV cache management as the root cause."（Abstract）；§3："The key insight is that the KV cache, produced by the shared VLM backbone from a common observation, is a natural locus for both computation reuse and execution coordination."
- 贡献：(1) 把多任务并行定为 MoT VLA 的目标推理场景，并指出隔离 KV cache 是低效根源；(2) 统一 KV cache 管理，支持跨任务 KV 共享与跨帧 continuous batching；(3) 在 π0.5 上实现，在 RTX 4090 与 Jetson AGX Thor 上评估并推广到其他骨干（§1 contributions）。
- 机制：同一模型内跨任务共享 prefill KV，跨帧把未完成的语言请求合批，每帧推进 k 个 token。
- 证据：相对隔离执行最多 3.7×，4090 上同时达到 >200 tokens/s 与 70 Hz（Abstract）；Thor 真机帧延迟 1030 → 393 ms（§4.7）。
- 对本场景：两路任务共享同一 VLM 和同一观测才能共享 prefill、合批 decode，而本场景执行器与规划器是两个独立模型。

### D28 FiS-VLA

- 问题：基于 VLM 的机器人策略执行频率低；已有双系统设计把两个系统做成独立模型，System 1 用不上 System 2 的预训练知识。
- 观察：论文动机为定性论述，没有给出动机测量（判断）；动作块 1 时实测 21.9 Hz，对比 CogACT 9.8、π0 13.8、OpenVLA 6.3 Hz（表 1）。
- 洞察："However, existing designs maintain both systems as separate models, limiting System 1 from fully leveraging the rich pretrained knowledge from the VLM-based System 2."（Abstract）
- 贡献：(1) 把 System 1 嵌入预训练 VLM 的统一双系统 VLA（FiS）；(2) 两系统采用异构模态输入与异步运行频率；(3) dual-aware co-training 联合优化两系统（§1 contributions）。
- 机制：System 1 复用 LLM 最后几层 transformer block（部分共享参数），训练时异步采样，使 System 2 以较低频率运行。
- 证据：仿真平均成功率比 SOTA 高 8%、真机高 11%，动作块 8 时 117.7 Hz（Abstract；附录 B.1 称之为 "theoretical control frequency"）；S2 : S1 = 1 : 4 最好（§4.2，仅图）。
- 对本场景：两系统共享参数、按固定比例交错，不存在两个独立模型在同一 GPU 上的并发与干扰。

### D29 RTC

- 问题：VLA 推理延迟高，action chunking 在块边界出现停顿或分布外的抖动动作。
- 观察：网络推理延迟 vanilla π0.5 76 ms、RTC 97 ms（表 1，附录 A.3）；同步推理随注入延迟线性退化（§4.2，图 6）。
- 洞察："Our key insight is to pose real-time chunking as an inpainting problem."（§3）
- 贡献：(1) diffusion / flow 动作块策略的异步实时推理系统；(2) 基于 Kinetix 的 12 个高动态任务新基准；(3) 用 π0.5 在 6 个真实双臂任务上评估，表明 RTC 对推理延迟稳健并提高任务吞吐（§1 contributions 段）。
- 机制：执行当前块时生成下一块，冻结推理延迟内必然执行的前 d 步，对其余步做 inpainting 引导。
- 证据：对注入延迟 "completely robust"，同步推理线性退化（§4.2，图 6）；同一真机动作比同步推理快 20%（图 1 说明）。
- 对本场景：推理在远端独占 GPU 上运行、没有共置，延迟被当作给定输入。

### D30 SmolVLA

- 问题：现有 VLA 参数量大、训练与部署成本高，且依赖学术 / 工业数据集，忽视社区采集的数据。
- 观察：论文没有测推理延迟；同步推理时机器人空等（§3.3，定性）。
- 洞察："In this work, we present SmolVLA, a small, efficient, and community-driven VLA that drastically reduces both training and inference costs, while retaining competitive performance."（Abstract）；异步部分："we introduce an asynchronous inference stack decoupling perception and action prediction from action execution, allowing higher control rates with chunked action generation."（Abstract）
- 贡献：(1) 轻量架构（跳层、少视觉 token、小 VLM、交替 cross / self-attention）；(2) 用不到 30k episode 的社区数据预训练；(3) 解耦动作执行与观测处理 / 动作预测的异步推理栈（§1 contributions）。
- 机制：动作队列剩余比例低于阈值 g 时发送新观测，g ≥ (E[ℓ_S]/Δt)/n 时可避免空等（§3.3）。
- 证据：异步相对同步任务完成时间 9.70 ± 2.95 s 对 13.75 ± 2.42 s，"around 30%" 更快；固定时间内 19 块对 9 块（§4.6，图 5）。
- 对本场景：单模型、可远端推理，没有第二个模型共置，也没有尾时延数据。
