# 经典 ML-systems / GPU-systems 论文如何论证核心贡献

检索日期：2026-11-17。正文中文，术语保留英文。引文为论文原文（从官方 PDF 用 `pdftotext` 抽出后逐字摘录），位置标注为"Abstract"、"§1 第 N 段"或图表编号。标 † 的内容未能在官方页面核实。标"判断"的内容是作者归纳，不是论文原话。

## 结论先行

10 篇论文从观察走到贡献，反复用的是 7 种论证动作（详见"归纳的模式"）：

| # | 模式 | 一句话 | 代表论文 |
|---|---|---|---|
| P1 | 换调度/管理的粒度 | 现有系统的调度原子与 workload 的自然边界不匹配 → 把原子改成那个边界 | Orca, Gandiva, vLLM |
| P2 | 借另一个领域的经典抽象 | "这个问题和 X 领域已解决的问题同构" → 把 X 的抽象搬过来，再处理不同之处 | vLLM, FlashAttention, AlpaServe, PipeDream |
| P3 | 把被假设为"不可预测"的量变成可预测并加以利用 | "现有系统默认 Y 不可预测 → 我们测量发现 Y 是确定的/周期的/可拟合的" | Clockwork, Gandiva, Pollux |
| P4 | 重新定义优化目标 | 现有指标（throughput、FLOPs）与真实收益错位 → 定义新指标（goodput、HBM 访问量），再按它优化 | Pollux, DistServe, FlashAttention |
| P5 | 拆开被耦合在一起的东西 | 两种特性不同的计算/资源被放在一起 → 互相干扰、配置被绑死 → 拆开后各自优化 | DistServe, Mooncake, Orca |
| P6 | 主动付出一种成本换另一种收益 | 明知会增加开销 A，但 A 远小于消除的瓶颈 B；论文用测量证明 A 小 | AlpaServe, FlashAttention, Mooncake, Clockwork |
| P7 | 用"加强版基线"和"拆掉质疑的测量"支撑论点 | 给基线 oracle 配置，或单独测出反方最可能质疑的开销 | vLLM, AlpaServe, Pollux, DistServe, Orca |

判断：几乎每篇都是"一张 motivating figure 量化与理想值的差距 → 一句可证伪的 insight → 一个按 insight 命名的机制 → 一个对齐 insight 的 headline 数字"。论文的 novelty 主要在第二步，即发现了一个现有系统没有利用的 workload 性质；后面的机制往往是该性质的直接推论。

---

## 选择标准与检索记录

### 选择标准

1. 会议：OSDI / SOSP / NSDI / FAST / NeurIPS，2018–2025。
2. 至少满足一项：(a) 官方页面标注的 Best Paper 或 Distinguished Artifact；(b) 高引用（Semantic Scholar 或 OpenAlex 计数在同年同领域中靠前）；(c) 开创了后续论文以它为基线的一条工作线。
3. 论证类型要覆盖：调度粒度、内存管理、可预测性、目标指标、解耦、IO 复杂度、集群调度。同一种论证只留最有代表性的一篇。
4. 必须能拿到官方全文 PDF（USENIX open access、ACM DOI 对应版本或 arXiv 正式版）。

### 入选 10 篇

| # | 论文 | Venue | 奖项（来源） | 引用数（来源，2026-11-17） | 入选理由 |
|---|---|---|---|---|---|
| 1 | vLLM / PagedAttention | SOSP'23 | 无。SOSP'23 program 页面的 3 个 Best Paper 标注分别属于 TreeSLS、JIT 编译器验证、Trio，没有本文 | 10,243 (S2) / 10,864 (OpenAlex) | 借 OS 抽象的典型 |
| 2 | Orca | OSDI'22 | 无（USENIX 论文页无奖项标注） | 未取得† | 换调度粒度的典型 |
| 3 | Clockwork | OSDI'20 | Distinguished Artifact Award（USENIX 论文页 "Distinguished Artifact Award Winner"） | 487 (S2) | 把不可预测变为可预测的典型 |
| 4 | AlpaServe | OSDI'23 | 无（USENIX 论文页无奖项标注） | 297 (S2) | 反直觉 trade-off 的典型 |
| 5 | PipeDream | SOSP'19 | 无。SOSP'19 官方 awards 页的两篇 Best Paper 是 thread-safety 检测和 Serval | 1,379 (S2) / 1,385 (OpenAlex) | 借流水线抽象，并处理训练特有的双向性 |
| 6 | Gandiva | OSDI'18 | 无（USENIX 论文页无奖项标注） | 670 (S2) | 利用 domain-specific 周期性 |
| 7 | FlashAttention | NeurIPS'22 | 无已核实的奖项。ICML'22 HAET workshop best paper † 未能在官方页面核实；ICML 页面只显示该 workshop 的 poster | 5,923 (S2) | IO-aware 论证：重定义代价模型 |
| 8 | DistServe | OSDI'24 | 无（USENIX 论文页无奖项标注） | 1,255 (S2) | 解耦 + goodput 指标 |
| 9 | Pollux | OSDI'21 | Jay Lepreau Best Paper（USENIX 论文页 "Awarded Best Paper!"） | 337 (S2，arXiv 条目) | 重定义目标指标 + 在线建模 |
| 10 | Mooncake | FAST'25 | Best Paper（USENIX 论文页 "Awarded Best Paper!"；奖项全称 "Erik Riedel Best Paper Award" 仅见于清华新闻转述†） | 329 (S2，arXiv 题名条目) / 345 (OpenAlex) | 工业部署，以存储换计算 |

### 考察后未入选

| 论文 | 原因 |
|---|---|
| Sarathi-Serve (OSDI'24) | 与 DistServe 面对同一个 prefill/decode 干扰问题，但给出相反的解（chunked prefill，保持 colocate）。作为 P1/P5 的对照写进"归纳的模式"，不单独做卡片。S2 引用 1,045 |
| TVM (OSDI'18) | 编译器类论证（learned cost model + 搜索），与本批的调度/服务主题距离较远。S2 引用 2,342（OSDI 条目） |
| Ray (OSDI'18) | 编程框架论证，不属于观察到贡献的 GPU 系统论证。S2 引用 1,874 |
| Shepherd (NSDI'23)、Llumnix (OSDI'24)、Tiresias (NSDI'19) | 论证模式与 Clockwork/AlpaServe、vLLM、Gandiva/Pollux 重复，且 USENIX 页面均无奖项标注（Shepherd、Llumnix 已查；Tiresias 未查†） |

### 检索记录

- 全文 PDF 下载自 usenix.org（Orca、Clockwork、AlpaServe、Gandiva、DistServe、Pollux、Mooncake、Sarathi-Serve、Shepherd）、arxiv.org（vLLM 2309.06180，FlashAttention 2205.14135v2），PipeDream SOSP'19 版取自 UW CSE550 课程镜像（版权页为 "SOSP '19 … https://doi.org/10.1145/3341301.3359646"）。arXiv 1806.03377 是 PipeDream 的早期版本，标题不同，未采用。
- 奖项：逐篇抓取 `usenix.org/conference/<conf>/presentation/<id>` 页面，搜索 "Awarded Best Paper!" / "Distinguished Artifact" 字样。SOSP'23 用 `sosp2023.mpi-sws.org/program.html` 中的 "Best Paper Award" 标注；SOSP'19 用 `sigops.org/s/conferences/sosp/2019/awards.html`。
- 引用数：Semantic Scholar Graph API（batch 与 `search/match`），OpenAlex `works?search=`。两者对 USENIX 论文的收录不稳定（如 Pollux 在 OpenAlex 仅 99，S2 对 Orca 连续限流），引用数只用于粗排，不作为结论依据。
- vLLM 的 arXiv v1 与 SOSP 版正文一致（arXiv 首页印有 SOSP'23 版权块与 DOI 10.1145/3600006.3613165）。

---

## 论文卡片

### 1. vLLM / PagedAttention

- **元数据**："Efficient Memory Management for Large Language Model Serving with PagedAttention"，Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica。SOSP'23，DOI 10.1145/3600006.3613165；arXiv 2309.06180。奖项：无。
- **一句话论点**（Abstract 第 3 句）："we propose PagedAttention, an attention algorithm inspired by the classical virtual memory and paging techniques in operating systems."
- **引出问题的观察**：
  - 吞吐受 batch size 限制，而 batch size 受 KV cache 内存限制。Fig. 1 左：13B 模型在 A100-40GB 上，参数占约 65%，KV cache 占 >30%（§1 第 3 段）。
  - 现有系统按最大长度预分配连续内存。Fig. 2："only 20.4% - 38.2% of the KV cache memory is used to store the actual token states in the existing systems"（§1，"First, the existing systems…" 段）。
  - KV cache 的特性："it dynamically grows and shrinks over time as the model generates new tokens, and its lifetime and length are not known a priori"（§1 "In this paper, we observe…" 段）。
- **关键洞察**：KV cache 的困境（动态增长、长度未知、需要共享）与 OS 进程内存的困境同构，所以可以把 OS 的解法搬过来："one can think of blocks as pages, tokens as bytes, and requests as processes"（§1 "To address the above limitations…" 段）。固定大小的 block 消除 external fragmentation，按需分配压低 internal fragmentation，block 粒度的 copy-on-write 让 parallel sampling / beam search 共享 KV。
- **贡献的类型**：借经典 OS 抽象（P2）+ 量化浪费（characterization）。管理粒度从"每请求一段连续内存"变为"固定大小 block"（P1）。
  - 论文自述贡献（§1 末尾列表）：
    - "We identify the challenges in memory allocation in serving LLMs and quantify their impact on serving performance."
    - "We propose PagedAttention, an attention algorithm that operates on KV cache stored in non-contiguous paged memory, which is inspired by the virtual memory and paging in OS."
    - "We design and implement vLLM, a distributed LLM serving engine built on top of PagedAttention."
    - "We evaluate vLLM on various scenarios and demonstrate that it substantially outperforms the previous state-of-the-art solutions such as FasterTransformer [31] and Orca [60]."
- **评估怎么对应论点**：
  - 指标选的是"同等 normalized latency 下可承受的请求率"（§6.1 "Key metrics"），直接对应"内存省下来 → batch 更大 → 吞吐更高"的因果链。
  - 基线：FasterTransformer，以及 Orca 的三种重新实现：Orca (Oracle)（已知真实输出长度）、Orca (Pow2)、Orca (Max)。Oracle 版是给基线开挂（P7）。
  - Headline（§6.2）："vLLM can sustain 1.7×–2.7× higher request rates compared to Orca (Oracle) and 2.7×–8× compared to Orca (Max)"；对 FasterTransformer "up to 22×"。
  - 机制解释用同一个量收尾：Fig. 13a，OPT-13B 上 vLLM 同时处理的请求数是 Orca (Oracle) 的 2.2×。观察（内存浪费）、机制、结果（并发数）用的是同一个变量。
  - 判断：Orca 不开源，基线是作者重新实现的，读者需自行评估实现差异。

### 2. Orca

- **元数据**："Orca: A Distributed Serving System for Transformer-Based Generative Models"，Yu, Jeong, Kim, Kim, Chun。OSDI'22，https://www.usenix.org/conference/osdi22/presentation/yu 。奖项：无。
- **一句话论点**（Abstract 第 4 句）："we propose iteration-level scheduling, a new scheduling mechanism that schedules execution at the granularity of iteration (instead of request) where the scheduler invokes the execution engine to run only a single iteration of the model on the batch."
- **引出问题的观察**：生成式模型的请求是多次迭代，而 serving 系统按请求调度。§3 C1："requests that have finished earlier than others in the batch cannot return to the client"；"when a new request arrives in the middle of the current batch's execution, the aforementioned scheduling mechanism makes the newly arrived request wait until all requests in the current batch have finished"。Fig. 3 用示意图说明，不是测量。
  - 判断：Orca 的问题陈述几乎全靠推理和示意图，没有 trace 统计。这在"新 workload 刚出现"的时间点可以成立，因为不匹配是结构性的，一画就懂。
- **关键洞察**：调度器与引擎只在"整批完成"时交互，这是继承自单次前向模型（ResNet、BERT）的假设；对 autoregressive 模型，自然边界是一次 iteration，所以调度原子应改为 iteration。第二个洞察针对随之出现的障碍（不同请求长度不同，attention 无法 batch）："Since the Attention operation is not associated with any model parameters, applying batching to Attention has no benefit of reducing the amount of GPU memory reads by reusing the loaded parameters across multiple requests"（§1 倒数第 3 段）。因此只对有权重的算子做 batch（selective batching）。
- **贡献的类型**：换调度粒度（P1）；把 attention 与其他算子解耦处理（P5 的局部形式）。论文 §1 没有逐条列出贡献，在 Abstract 中以"iteration-level scheduling + selective batching + 分布式设计"三件事陈述。
- **评估怎么对应论点**：
  - §6.1 engine microbenchmark 先证明 selective batching 的代价小：所有请求等长时 Orca engine 与 FasterTransformer "similar (or slightly worse)"。这是提前拆掉"不 batch attention 会变慢"这个质疑（P7）。
  - 端到端：GPT-3 规模模型，合成 trace。Headline（§6.2）："to match a median normalized latency of 190ms for the 175B model … FasterTransformer provides a throughput of 0.185 req/s whereas ORCA provides a throughput of 6.81 req/s, which is a 36.9× speedup."
  - 收益来源归因到 C1："the ORCA scheduler makes late-arrived requests hitch a ride with the current ongoing batch"。

### 3. Clockwork

- **元数据**："Serving DNNs like Clockwork: Performance Predictability from the Bottom Up"，Gujarati, Karimi, Alzayat, Hao, Kaufmann, Vigfusson, Mace。OSDI'20，https://www.usenix.org/conference/osdi20/presentation/gujarati 。奖项：OSDI'20 Distinguished Artifact Award（USENIX 论文页）。
- **一句话论点**（Abstract 第 3–4 句）："Yet the underlying execution times are not fundamentally unpredictable—on the contrary we observe that inference using Deep Neural Network (DNN) models has deterministic performance."
- **引出问题的观察**：
  - 现状假设（§1 第 3 段）："Existing systems fundamentally assume that the constituent system components have unpredictable latency performance"；而且 best-effort 机制会 "cascade the unpredictability to other system components"。
  - 反例测量（§2 "Observation: DNN inference is predictable"，Fig. 2a）：V100、batch size 1 单独运行时，"The 99.99th percentile latency was within 0.03% of the median latency."
  - 并发的代价（Fig. 2 caption）："Running inferences concurrently gains up to 25% throughput (middle), at a cost of substantially increased latency variability (right)"。
- **关键洞察**：DNN 推理没有条件分支，是确定的计算序列，所以执行时间可预测。只要系统中每一层都不引入额外的不确定性，就能提前算出每个请求能否满足 SLO，从而用 proactive 调度取代 reactive 调度。方法论叫 "consolidating choice"：去掉所有 best-effort 机制，把调度决策集中到 controller。
- **贡献的类型**：把被假设为不可预测的量变成可预测（P3）；提出设计原则（design methodology）；为了可预测性牺牲部分吞吐（P6：worker 一次只跑一个 EXEC，"reduces performance variability by two orders of magnitude while only minimally decreasing inference throughput"，§4.4）。
  - 论文自述贡献（§1 末尾列表）：
    - "We demonstrate that predictability is a fundamental trait of DNN inference that can be exploited to build a predictable model serving system."
    - "We propose a system design approach, consolidating choice, to preserve predictable responsiveness in a larger system comprised of components with predictable performance."
    - "We present the design and implementation of Clockwork, a distributed model serving system that mitigates tail latency of DNN inference from the bottom up."
    - "We report from an experimental evaluation on Clockwork to show that the system supports thousands of models concurrently per GPU and substantially mitigates tail latency, even while supporting tight latency SLOs. …"
- **评估怎么对应论点**：
  - 基线 Clipper、INFaaS。指标是 goodput（SLO 内完成的请求数）而非吞吐，直接对应"可预测 → 能守住 SLO"的论点。
  - §6.1 Fig. 5：SLO 从 10 ms 扫到 500 ms。500 ms 时三者 goodput 相当（约 800 r/s），SLO 收紧后 "goodput collapses below a 100 ms SLO" 发生在 Clipper 和 INFaaS 上，"only Clockwork can continue serving SLOs below 100 ms"。
  - 规模与真实负载：单 worker 3,600 个模型（Fig. 6）；Microsoft Azure Functions trace（§6.5，Fig. 9）。Abstract headline："meeting 100 ms latency targets for 99.9999% of requests"。
  - 判断：Clockwork 的评估刻意扫 SLO 紧度，因为论点是"紧 SLO 下只有可预测系统能活下来"，扫这个轴最能拉开差距。

### 4. AlpaServe

- **元数据**："AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving"，Li, Zheng, Zhong, Liu, Sheng, Jin, Huang, Chen, Zhang, Gonzalez, Stoica。OSDI'23，https://www.usenix.org/conference/osdi23/presentation/li-zhuohan 。奖项：无（页面仅有导航栏 "Best Papers" 链接，无本文标注）。
- **一句话论点**（Abstract 第 1–2 句）："Model parallelism is conventionally viewed as a method to scale a single large deep learning model beyond the memory limits of a single device. In this paper, we demonstrate that model parallelism can be additionally used for the statistical multiplexing of multiple devices when serving multiple models, even when a single model can fit into a single device."
- **引出问题的观察**：
  - 负载突发（§1 第 3 段）："using common workload traces, we observe frequent spikes in demand of up to 50× the average"。按峰值预留导致设备大部分时间闲置。
  - 现有做法的默认假设（§1 第 5 段）："the natural approach assumed by almost all existing serving systems [9, 33, 34] is to allocate one dedicated GPU for one model"，理由是切分模型会增加通信开销。
  - Fig. 1 算例：2 模型 2 GPU，独占放置时 4 个突发请求的平均延迟 2.5y；切成模型并行并共置（假设 10% 开销）后为 1.85y。
- **关键洞察**：模型并行会让单请求变慢，但让每个模型的突发能用上所有 GPU，即 statistical multiplexing（借自网络领域）。只要负载足够突发，multiplexing 收益大于并行开销。"inducing additional model parallelism (to the point where per-example execution time actually increases) enables a wider range of placement strategies"（§1 第 5 段）。
- **贡献的类型**：反直觉 trade-off（P6）；借网络的 statistical multiplexing 概念（P2）；trade-off 空间的 characterization。
  - 论文自述贡献（§1 末尾列表）：
    - "A detailed analysis of the tradeoff space of different model parallel strategies for efficient model serving."
    - "Novel model placement algorithms to incorporate model parallelism in a serving system."
    - "A comprehensive evaluation of AlpaServe with both synthetic and production workloads."
- **评估怎么对应论点**：
  - §3 先做 trade-off 分析：何时并行有利，与 burstiness、per-GPU 内存、SLO 的关系。这对应贡献 1。
  - 端到端（§6.2）：Azure Functions trace MAF1/MAF2，主要在模拟器上跑。论文先报告模拟器与真实系统的误差 "less than 2% in all cases"，并在 64-GPU 集群上做部分实验。
  - 基线：Selective Replication（同一放置算法，去掉模型并行，用来隔离"模型并行"这一变量）；Clockwork++（"assuming zero swapping overheads"，作者称其为 "a hypothetical upper bound of Clockwork's performance"）。两个基线分别是消融式基线和加强版基线（P7）。
  - Headline（Abstract）："up to 10× higher rates or 6× more burstiness while staying within latency constraints for more than 99% of requests"。评估按 rate、burstiness、SLO scale、cluster size 分别扫，每个轴对应 trade-off 分析里的一个因素。

### 5. PipeDream

- **元数据**："PipeDream: Generalized Pipeline Parallelism for DNN Training"，Narayanan, Harlap, Phanishayee, Seshadri, Devanur, Ganger, Gibbons, Zaharia。SOSP'19，DOI 10.1145/3341301.3359646。奖项：无。
- **一句话论点**（Abstract 第 3 句）："We present PipeDream, a system that adds inter-batch pipelining to intra-batch parallelism to further improve parallel training throughput, helping to better overlap computation with communication and reduce the amount of communication when possible."
- **引出问题的观察**（§1 第 2 段，Fig. 1）：data parallelism 的通信开销，"Over 32 GPUs, the communication overhead for some models, computed as the percentage of total time spent on communication stalls, is as high as 90% due to expensive cross-server all_reduce communication"；并给出趋势论证："rapid increases in GPU compute capacity over time will further shift the bottleneck of training towards communication for all models."
- **关键洞察**：按层切分后，跨 worker 只需传相邻层之间的 activation/gradient，通常远小于整个模型，且是点对点而非 all-to-all。经典流水线不能直接套用，因为 "DNN training is bi-directional"（§1 第 4 段）；朴素做法要么牺牲 statistical efficiency，要么频繁 flush（GPipe）。PipeDream 的解法：1F1B 交替调度消除稳态 stall；weight stashing 保证 backward 用与 forward 相同的权重版本，"learning is still effective because model weights change relatively slowly and bounded staleness has been found effective"（§1 第 5 段）。
- **贡献的类型**：借经典抽象（处理器流水线），并找出训练与经典流水线的关键不同（双向、有状态）（P2）；用额外内存（多份权重）换掉 pipeline flush（P6）；同时考虑 hardware efficiency 和 statistical efficiency 两个目标（P4 的雏形，Pollux 后来把它形式化）。论文 §1 无逐条贡献列表，贡献以 1F1B、weight stashing、自动切分三部分在正文中陈述。
- **评估怎么对应论点**（§1 最后一段汇总）："Compared to data-parallel training, PipeDream reaches a high target accuracy on multi-GPU machines up to 5.3× faster for image classification tasks, up to 3.1× faster for machine translation tasks, 4.3× faster for language modeling tasks, and 3× faster for video captioning models. PipeDream is also 2.6× – 15× faster than model parallelism, up to 1.9× faster than hybrid parallelism, and 1.7× faster than simpler approaches to pipelining such as GPipe's approach."
  - 指标是 time-to-target-accuracy，而不是 samples/s。这正好回应"weight stashing / staleness 会不会伤精度"的质疑（P7）。
  - 基线覆盖论文提到的每一种替代：DP、MP、hybrid、GPipe 式流水线，每个基线对应 §1 中被否定的一种方案。

### 6. Gandiva

- **元数据**："Gandiva: Introspective Cluster Scheduling for Deep Learning"，Xiao, Bhardwaj, Ramjee, Sivathanu, Kwatra, Han, Patel, Peng, Zhao, Zhang, Yang, Zhou。OSDI'18，https://www.usenix.org/conference/osdi18/presentation/xiao 。奖项：无。
- **一句话论点**（Abstract 第 1 句）："We introduce Gandiva, a new cluster scheduling framework that utilizes domain-specific knowledge to improve latency and efficiency of training deep learning models in a GPU cluster."；核心机制句（Abstract）："Gandiva addresses these two challenges by exploiting a third key characteristic of deep learning: intra-job predictability, as they perform numerous repetitive iterations called mini-batch iterations."
- **引出问题的观察**：
  - 现状（§1 第 3 段）：DL 作业被当作 "yet another big-data job that is allocated a set of GPUs at job startup and holds exclusive access to its GPUs until completion"。
  - workload 特性一：feedback-driven exploration（超参搜索需要所有子作业同时给出早期反馈），与独占式排队冲突。
  - workload 特性二：异构，对 affinity 和干扰的敏感度不同（Fig. 1–4）。
  - 周期性测量（§3，Fig. 5a）：ResNet-50 在 4×K80 上，每个 mini-batch 约 1.5 s 为一个周期，"The maximum and minimum GPU memory used is 23GB and 0.3GB, respectively, or a factor of 77x."
- **关键洞察**：DL 作业由数百万个形状相同、边界清晰的 mini-batch 组成，所以 (1) 可以在内存低谷处 suspend/migrate，代价极小 → time-slicing 与迁移变得实用；(2) mini-batch 进度率是可靠的性能探针 → 调度器可以试错并纠正（introspection）。"in effect, it re-defines the atom of scheduling from a job to automatically-partitioned micro-tasks"（§1 第 6 段）。
- **贡献的类型**：利用 domain-specific 可预测性（P3）；换调度原子，从 job 到 mini-batch 边界上的 micro-task（P1）；打开黑盒，scheduler 与框架协同设计。
  - 论文自述贡献（§1 末尾列表）：
    - "We illustrate various unique characteristics of the deep learning workflow and map it to specific requirements needed for cluster scheduling."
    - "We identify generic primitives that can be used by a DLT job scheduling policy, and provide application-aware techniques to make primitives such as time-slicing and migration an order of magnitude more efficient and thus practical by leveraging DL-specific knowledge of intra-job periodicity."
    - "We propose and evaluate a new introspective scheduling framework that utilizes domain-specific knowledge of DLT jobs to refine its scheduling decision continuously, thereby significantly improving early feedback time and delivering high cluster efficiency."
- **评估怎么对应论点**：
  - 两个痛点各有一个指标。Early feedback：到 100 个 mini-batch 的平均时间，"498s for Gandiva and 2,203s for the baseline, for a reduction of 77%"（§6.3）。Efficiency：180-GPU 集群真实负载，cluster utilization 提升 26%（Abstract）。
  - 原语开销单独测（Fig. 11–13，迁移、grow/shrink），对应贡献 2 中"order of magnitude more efficient"的说法。
  - 超参搜索端到端："up to an order of magnitude"（Abstract）。

### 7. FlashAttention

- **元数据**："FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"，Dao, Fu, Ermon, Rudra, Ré。NeurIPS 2022，https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html ；arXiv 2205.14135。奖项：无已核实的奖项（ICML'22 HAET workshop best paper †）。
- **一句话论点**（Abstract 第 3 句）："We argue that a missing principle is making attention algorithms IO-aware—accounting for reads and writes between levels of GPU memory."
- **引出问题的观察**：
  - 已有方法的失败（§1 第 2 段）：approximate attention 把 FLOPs 降到线性，"many of them do not display wall-clock speedup against standard attention and have not gained wide adoption. One main reason is that they focus on FLOP reduction (which may not correlate with wall-clock speed) and tend to ignore overheads from memory access (IO)."
  - 硬件趋势（§1 第 4 段）："On modern GPUs, compute speed has out-paced memory speed [61, 62, 63], and most operations in Transformers are bottlenecked by memory accesses"。
  - 直接测量（§3.2，Fig. 2 左，GPT-2 medium，A100）：standard attention 66.6 GFLOPs、40.3 GB HBM R/W、41.7 ms；FlashAttention 75.2 GFLOPs、4.4 GB、7.3 ms。原文："even though FlashAttention has higher FLOP count compared to standard attention (due to recomputation in the backward pass), it has much fewer HBM accesses, resulting in much faster runtime."
- **关键洞察**：attention 的真实代价是 HBM 读写而不是 FLOPs，所以只要不把 N×N 矩阵写回 HBM，即使多算（backward 重算）也会更快。实现靠两个 "well-established techniques"：tiling（softmax 增量归约）和 recomputation（§1 第 5 段）。
- **贡献的类型**：重定义代价模型（FLOPs → IO complexity）（P4）；借另一个领域的经典分析框架，即 IO complexity / external memory model，原文列举 database joins、image processing、numerical linear algebra（P2）；用更多计算换更少访存（P6）。另有理论贡献：上界 O(N²d²M⁻¹)，以及 "no exact attention algorithm can asymptotically improve on the number of HBM accesses over all SRAM sizes"（§1 第 6 段）。
  - 论文自述的实证贡献（§1 末尾列表，节选首句）：
    - "Faster Model Training. FlashAttention trains Transformer models faster in wall-clock time."
    - "Higher Quality Models. FlashAttention scales Transformers to longer sequences, which improves their quality and enables new capabilities."
    - "Benchmarking Attention. FlashAttention is up to 3× faster than the standard attention implementation across common sequence lengths from 128 to 2K and scales up to 64K."
- **评估怎么对应论点**：
  - 论点"IO 决定运行时间"有专门的验证实验：Fig. 2 中图改变 block size 以改变 HBM 访问量，"As block size increases, the number of HBM accesses decreases … and runtime decreases"（§3.2）。这是先验证代价模型，再报告加速。
  - 端到端：BERT-large 比 MLPerf 1.1 训练纪录快 15%；GPT-2 比 HuggingFace / Megatron-LM 快 3×；LRA 快 2.4×（Abstract）。
  - "exact" 的价值通过质量结果体现：更长 context 带来 GPT-2 perplexity 改善 0.7，Path-X（16K）61.4%（Abstract）。
  - 判断：论文先证明代价模型正确（Fig. 2），再证明在该模型下最优（下界），最后给出 wall-clock 结果。这条论证链比单纯报告加速更难被反驳。

### 8. DistServe

- **元数据**："DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving"，Zhong, Liu, Chen, Hu, Zhu, Liu, Jin, Zhang。OSDI'24，https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin 。奖项：无。
- **一句话论点**（Abstract 第 2–3 句）："Existing LLM serving systems colocate the two phases and batch the computation of prefill and decoding across all users and requests. We find that this strategy not only leads to strong prefill-decoding interferences but also couples the resource allocation and parallelism plans for both phases."
- **引出问题的观察**（§1 第 5 段，Fig. 1）：13B 模型、A100、输入 512 / 输出 64，90% SLO attainment 下，现有系统单卡 goodput "about 1.6 requests per second (rps)"；只跑 prefill 的 GPU 为 5.6 rps，只跑 decode 的为 10 rps；"by allocating 2 GPUs for prefill and 1 GPU for decoding, we can effectively serve the model with an overall goodput of 10 rps, or equally 3.3 rps per GPU, which is 2.1x higher than existing systems."
- **关键洞察**：prefill（compute-bound，看 TTFT）与 decode（memory-bound，看 TPOT）特性和 SLO 都不同，放在同一 GPU 上既互相干扰，又只能共用一种并行配置。拆到不同 GPU 后，两个问题同时消失，代价是 KV cache 传输；作者论证该代价在现代集群中很小（§3.3）。
- **贡献的类型**：解耦（P5）；重新定义指标为 per-GPU goodput，即满足 TTFT 与 TPOT 双 SLO 的最大请求率（P4）；用 KV 传输换消除干扰（P6）。
  - 论文自述贡献（§1 末尾列表）：
    - "Identify the problems of prefill-decoding interference and resource coupling in existing LLM serving systems and propose to disaggregate the two phases."
    - "Design a novel placement algorithm to choose the goodput-optimal schema for prefill and decoding instances automatically."
    - "Conduct a comprehensive evaluation of DistServe with realistic workloads."
- **评估怎么对应论点**：
  - 基线：vLLM（代表 colocate + continuous batching）和 DeepSpeed-MII（chunked prefill，"It mitigates but cannot eliminate the prefill-decoding interference"）。前者对应"不处理干扰"，后者对应"缓解但不拆开"，两个基线分别否定一种替代方案。
  - Headline（Abstract）："DistServe can serve 7.4× more requests or 12.6× tighter SLO, compared to state-of-the-art systems, while staying within latency constraints for > 90% of requests." 对 DeepSpeed-MII 为 1.6×–7.4×（§6.2）。
  - 拆掉主要质疑（P7）：§6.3 latency breakdown，OPT-175B 上 "the KV Cache transmission only accounts for less than 0.1% of the total latency"，"over 95% of requests experience a delay of less than 30ms"。
  - 放置算法依赖模拟器（§4），判断：与 AlpaServe 同一作者群，方法论一脉相承（simulator-guided placement）。

### 9. Pollux

- **元数据**："Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning"，Qiao, Choe, Subramanya, Neiswanger, Ho, Zhang, Ganger, Xing。OSDI'21，https://www.usenix.org/conference/osdi21/presentation/qiao 。奖项：OSDI'21 Jay Lepreau Best Paper Award（USENIX 论文页 "Awarded Best Paper!"）。
- **一句话论点**（Abstract 第 1 句）："Pollux improves scheduling performance in deep learning (DL) clusters by adaptively co-optimizing inter-dependent factors both at the per-job level and at the cluster-wide level."
- **引出问题的观察**：
  - 现状（Abstract 第 2–3 句）："Most existing schedulers expect users to specify the number of resources for each job … Some recent schedulers choose job resources for users, but do so without awareness of how DL training can be re-optimized to better utilize the provided resources."
  - 两个量此消彼长（§1 第 5–6 段，Fig. 1，ResNet18/CIFAR-10）：增大 batch size 提高 system throughput 与可扩展性，但降低 statistical efficiency；最优 batch size 依赖分到的 GPU 数和训练阶段，"A job in a later stage of training can potentially tolerate 10x or larger batch sizes without degrading statistical efficiency, than earlier during training"。
- **关键洞察**：资源量、batch size、学习率互相依赖，且随时间和集群负载变化，所以必须由调度器与作业联合、在线地决定。可行的前提是两个量都能在训练中被观测和建模：throughput 可以拟合，statistical efficiency 可以用 gradient noise scale 估计。于是定义 GOODPUT = THROUGHPUT × EFFICIENCY（§3，式 GOODPUT_t(⋆) = THROUGHPUT(⋆) × EFFICIENCY_t(M(⋆))），按它分配资源。
- **贡献的类型**：重定义目标指标（P4）；把原本由用户凭经验配置的量变成在线可预测的模型（P3）；把调度器与作业训练参数协同优化（co-adaptive）。
  - 论文自述贡献（§1 列表，⋆ 项）：
    - "We propose a formulation of goodput for DL jobs, which is a holistic measure of training performance that takes into account both system throughput and statistical efficiency."
    - "We show that a model of a DL job's goodput can be learned by observing its throughput and statistical behavior during training, and used for predicting the performance given different resource allocations and batch sizes."
    - "We design and implement a scheduling architecture that uses such models to configure the right combination of resource allocation and training parameters for each pending and running DL job. …"
    - "We evaluate Pollux on a cluster testbed using a workload derived from a Microsoft cluster trace. … Even when all jobs are manually tuned beforehand, Pollux reduces the average job completion time by 37%–50%. …"
    - "We show that, in cloud environments, using goodput-driven auto-scaling based on Pollux can potentially reduce the cost of training large models by 25%."
- **评估怎么对应论点**：
  - 基线 Tiresias、Optimus。关键设计是给基线理想配置（P7）："even when they are provided with ideal resource and training configurations for every job"（Abstract），这时仍减少 JCT 37–50%。如果只对比用户手配的情况，提升为 "up to 73%"（§1）。两个数字分开报告，前者证明增益来自 co-adaptation 而不是"帮用户调参"。
  - 公平性（finish-time fairness 1.5×–5.4×）和云成本（25%）对应 goodput 指标的两个延伸用途。
  - 真实集群 + trace-driven 模拟两层验证。

### 10. Mooncake

- **元数据**："Mooncake: Trading More Storage for Less Computation — A KVCache-centric Architecture for Serving LLM Chatbot"，Qin, Li, He, Cui, Ren, Zhang, Wu, Zheng, Xu。FAST'25，https://www.usenix.org/conference/fast25/presentation/qin 。奖项：FAST'25 Best Paper（USENIX 论文页 "Awarded Best Paper!"）。
- **一句话论点**：标题即论点 "Trading More Storage for Less Computation"；Abstract 第 2 句："This platform features a KVCache-centric disaggregated architecture that not only separates prefill and decoding clusters but also efficiently utilizes the underexploited CPU, DRAM, SSD and NIC resources of the GPU cluster to establish a disaggregated KVCache."
- **引出问题的观察**：
  - 现有 prefix caching 的局限（§2.2 "More Storage for Less Computation"）："existing approaches [16–18] typically restrict caching to local HBM and DRAM, assuming that the transfer bandwidth required for global scheduling would be prohibitively high. But, as we will described later in §5.3, the capacity of local DRAM supports only up to 50% of the theoretical cache hit rate"。
  - 资源闲置（§1 第 3 段）：GPU 节点上的 CPU、DRAM、SSD、RDMA 资源未被充分利用。
  - 带宽门槛的解析计算（§2.2 式 2）：LLaMA3-70B、8×A800、prefix 8192 时，复用 KV 比重算更快所需的加载带宽 B 至少为 6 GB/s，8×H800 为 19 GB/s；"a fully utilized 100 Gbps NIC per NVIDIA A800 HGX network is sufficient to meet these criteria"。
- **关键洞察**：被普遍假设为"太贵"的跨节点 KV 传输，算出来其实低于重算 prefill 的代价，而且模型越大越划算（d 越大越容易满足）。所以可以把集群里闲置的 DRAM/SSD 池化为全局 KV cache，再以 KVCache 为中心做调度（选 prefill 节点时同时考虑负载与 prefix 命中长度）。
- **贡献的类型**：以存储换计算（P6）；解耦（P5），既拆 prefill/decode，也拆 KV cache 与 GPU；推翻"带宽不够"这一隐含假设（P3 的变体：把被假设为过高的代价算出来）。论文 §1 无逐条贡献列表；以 Mooncake Store、KVCache-centric 调度（含热点复制）、chunked pipeline parallelism、transfer engine 四部分陈述。另外，它在 §1 明确回应 P/D 分离的争论："there are currently debates on its feasibility in large-scale practice due to bandwidth requirements and trade-offs associated with chunked prefill (e.g., Sarathi-Serve [10])"。
- **评估怎么对应论点**：
  - 生产数据（Abstract）："Mooncake is operational across thousands of nodes, processing over 100 billion tokens daily … enables Kimi to handle 115% and 107% more requests on NVIDIA A800 and H800 clusters, respectively, compared to previous systems." 由于模型是专有的，可复现实验改用 "a dummy model mirroring the architecture of LLaMA3-70B, based on replayed traces of actual workloads"（§1），并开源 trace。
  - 基线：vLLM、vLLM Prefix Caching、vLLM Chunked Prefill（Fig. 1），分别对应"无缓存"、"本地缓存"、"不分离而用 chunked prefill"三种替代方案。Headline："increases the effective request capacity by 59%∼498%"（Abstract）。
  - 论点"全局缓存优于本地缓存"的直接证据（§5.3）：cache hit rate 最高为本地缓存的 2.36×，prefill 计算时间最多节省 48%。
  - 论点"带宽够用"的证据（§5.4）：transfer engine 比现有方案快约 2.4× 与 4.6×。

---

## 归纳的模式

以下 7 种是"从观察到贡献"的论证动作。一篇论文通常叠加 2–3 种。每个模式给出结构、例子，以及使用时论文必须提供的证据（判断）。

### P1 换调度/管理的粒度

**结构**：现有系统的调度或分配原子（request、job、contiguous buffer）是从上一代 workload 继承的 → 新 workload 有更细的自然边界（iteration、mini-batch、token block） → 在新边界上调度，从而获得 preemption、插队、共享等能力 → 处理改粒度带来的新障碍。

| 论文 | 旧原子 → 新原子 | 改粒度后的新障碍及解法 |
|---|---|---|
| Orca | request → iteration | 不同长度请求的 attention 无法 batch → selective batching（attention 无权重，不 batch 损失小） |
| Gandiva | job → mini-batch 边界上的 micro-task | suspend 代价 → 在内存低谷（最小 0.3 GB vs 最大 23 GB）处切换 |
| vLLM | 每请求一段最大长度的连续内存 → 固定大小的 KV block | attention kernel 需读非连续内存 → PagedAttention kernel |
| 对照：Sarathi-Serve | 整个 prefill → 等大小的 prefill chunk | 与 decode 混批，"stall-free schedules"（Abstract） |

判断：这种论证的说服力来自"自然边界"本身显而易见（autoregressive 的 iteration、SGD 的 mini-batch），所以 Orca 只用示意图就能立论。代价是论文必须用实验说明新粒度的开销小（Orca §6.1 microbenchmark）。

### P2 借另一个领域的经典抽象

**结构**：指出新问题与某领域已解决问题同构（"one can think of blocks as pages, tokens as bytes, and requests as processes"）→ 把抽象搬过来 → 重点论述"本问题与原领域的不同之处"，这部分是真正的 novelty。

| 论文 | 借来的抽象 | 来源领域 | 论文强调的不同之处 |
|---|---|---|---|
| vLLM | virtual memory + paging + copy-on-write | OS | 需要定制 attention kernel；共享发生在 beam/parallel sampling 的序列之间 |
| FlashAttention | IO complexity、tiling | 外存算法 / DB join / 数值线性代数 | softmax 需要全行归约 → 在线 softmax；backward 用 recomputation |
| AlpaServe | statistical multiplexing | 网络 | 复用的代价是模型并行开销，不是丢包或排队 |
| PipeDream | pipelining | 处理器体系结构 | "DNN training is bi-directional" → 1F1B + weight stashing |

判断：评审最可能问"这不就是 X 吗"。这类论文在 §1 就主动承认借鉴（vLLM 标题级别承认，FlashAttention 称 "two well-established techniques"），然后把篇幅花在不同之处上。主动承认比被评审指出更有利。

### P3 把被假设为"不可预测"或"太贵"的量测出来，发现它可预测/可负担，再加以利用

**结构**：明确写出现有系统的隐含假设（"Existing systems fundamentally assume…"）→ 一个干净的测量推翻它 → 在新前提下，以前不可行的设计（proactive 调度、试错式调度、全局缓存）变得可行。

| 论文 | 被推翻的假设 | 推翻它的测量 | 由此可行的设计 |
|---|---|---|---|
| Clockwork | 组件延迟不可预测 | p99.99 与中位数相差 0.03%（Fig. 2a） | 集中式 proactive 调度，按 SLO 准入 |
| Gandiva | DL 作业是黑盒 | 内存随 mini-batch 呈 77× 周期波动（Fig. 5a） | 低代价 time-slicing / 迁移 + introspection |
| Pollux | 资源/batch size 只能由用户凭经验配置 | throughput 与 gradient noise scale 可在训练中在线拟合 | goodput 模型驱动的联合配置 |
| Mooncake | 跨节点 KV 传输带宽要求过高 | 式 2：70B 模型只需 6–19 GB/s，100 Gbps NIC 足够 | 全局分布式 KV cache 池 |

判断：这是本批中奖项最集中的模式（Pollux 与 Mooncake 获 Best Paper，Clockwork 获 Distinguished Artifact）。它要求论文给出一个对假设的直接测量，而不是间接的端到端改善。

### P4 重新定义优化目标或代价模型

**结构**：现有系统优化的指标（throughput、FLOPs、samples/s）与用户真实收益错位 → 定义新指标，把被忽略的因素乘进去 → 新指标下的最优解与旧指标不同 → 系统按新指标搜索或调度。

| 论文 | 旧指标 | 新指标 | 新指标改变了什么决策 |
|---|---|---|---|
| Pollux | system throughput | goodput = throughput × statistical efficiency | batch size 与 GPU 数随训练阶段变化 |
| DistServe | 整体 token 吞吐 | per-GPU goodput（同时满足 TTFT 与 TPOT SLO 的最大请求率） | 拆分 P/D，两阶段用不同并行配置 |
| FlashAttention | FLOPs | HBM 读写量（IO complexity） | 接受更多 FLOPs（重算）以换取更少 IO |
| Clockwork | 吞吐 / 平均延迟 | goodput（SLO 内完成数）、SLO 紧度 | 宁可降低吞吐也不并发执行 |

判断：新指标通常作为评估的主轴出现，从而决定实验怎么画图（DistServe 和 Clockwork 都扫 SLO 紧度，Pollux 报 JCT）。指标一旦被社区接受（"goodput"在 LLM serving 中已成通用词），就会成为后续论文的默认评价方式，这也是这类论文影响力大的原因之一。

### P5 拆开被耦合在一起的东西

**结构**：两种特性不同的计算或资源被放在同一单元上 → 指出两种代价：互相干扰，以及配置被绑定（只能选一个折中） → 拆开后各自优化 → 证明拆开的通信代价小。

| 论文 | 被拆开的对象 | 拆开的代价及其测量 |
|---|---|---|
| DistServe | prefill 与 decode 的 GPU 及并行配置 | KV 传输 <0.1% 总延迟（§6.3） |
| Mooncake | prefill/decode 集群；KV cache 与 GPU HBM | 带宽门槛解析（§2.2）+ transfer engine 测量（§5.4） |
| Orca（局部） | attention 与有权重算子的 batch 方式 | engine microbenchmark 中与 FasterTransformer 相当（§6.1） |

对照：Sarathi-Serve 面对同一干扰问题选择不拆，而是改粒度（P1）。Mooncake 在 §1 点名这场争论，并以生产数据说明"P/D disaggregation is preferable for scenarios with stringent SLO limits"。判断：同一个观察可以导向 P1 或 P5 两种贡献，取决于作者认为通信和资源是否充裕。

### P6 主动付出一种成本，换取更大的收益

**结构**：提出一个表面上"更差"的选择 → 用算例或测量说明付出的成本 A 远小于消除的瓶颈 B → 在什么条件下成立（边界条件）也一并给出。

| 论文 | 主动付出的成本 | 换来的收益 | 论文给出的边界条件 |
|---|---|---|---|
| AlpaServe | 模型并行开销（单请求更慢） | 突发负载下的 statistical multiplexing | burstiness、per-GPU 内存、SLO（§3） |
| FlashAttention | 更多 FLOPs（66.6 → 75.2 GFLOPs） | HBM 读写 40.3 → 4.4 GB，时间 41.7 → 7.3 ms | block size 超过 256 后转为受算力限制（§3.2） |
| Mooncake | 存储与网络 | 省去 prefill 重算 | 式 2：B 相对 G 的门槛，模型越大越易满足 |
| Clockwork | 放弃并发（最多 25% 吞吐） | 延迟方差降两个数量级 | 紧 SLO 场景 |
| PipeDream | 多份权重的内存（weight stashing） | 无 pipeline flush，同时保持 statistical efficiency | 限制 in-flight minibatch 数 |

判断：这类论文的 Fig. 1 常是 toy example 的算术（AlpaServe 的 2.5y vs 1.85y，DistServe 的 1.6 vs 3.3 rps），让读者在第一页就算出反直觉结论成立。边界条件写清楚后，"在某些场景更差"就从弱点变成了设计空间分析。

### P7 用加强版基线和拆掉质疑的测量支撑论点

**结构**：评估不仅证明"我们更快"，还要排除"更快是因为别的原因"。常用两种手法：

1. **加强版 / 消融式基线**：给基线完美信息或去掉自身唯一的新变量。
   - vLLM：Orca (Oracle) 已知真实输出长度。
   - AlpaServe：Clockwork++ 假设零 swap 开销，作者称其为 Clockwork 的 "hypothetical upper bound"；Selective Replication 是"去掉模型并行的 AlpaServe"。
   - Pollux：基线拿到每个作业的理想配置后，仍差 37–50%。
2. **对最可能的质疑单独测量**：
   - Orca："不 batch attention 会变慢吗" → 等长请求 microbenchmark。
   - DistServe："拆开后传输会不会很贵" → <0.1%。
   - FlashAttention："多算会不会更慢" → 改变 block size 验证 IO 决定运行时间。
   - PipeDream："staleness 会不会伤精度" → 用 time-to-accuracy 而非吞吐作为主指标。

判断：工作线之间存在明确的基线接力：Clockwork → AlpaServe（Clockwork++），Orca → vLLM（Orca 三种变体）→ DistServe（vLLM）→ Mooncake（vLLM 三种配置），Gandiva → Tiresias → Pollux（Tiresias）。每篇论文把前一篇定义为"现有系统"，并用前一篇没有利用的 workload 性质立论。

### 附：从本批论文抽出的论证模板（判断）

1. 一句话写出现有系统的隐含假设（粒度、可预测性、指标、耦合、代价）。
2. 一张图量化该假设造成的损失，最好以"实际值 vs 理想上界"的形式给出（vLLM 20.4–38.2% 利用率；DistServe 1.6 vs 3.3 rps；PipeDream 90% 通信 stall；FlashAttention 40.3 vs 4.4 GB）。
3. 一句 "X is true, so we can Y" 的 insight，X 是一个可单独验证的 workload 或硬件性质。
4. 机制以 insight 命名（iteration-level scheduling、PagedAttention、consolidating choice、goodput、disaggregation）。
5. 评估的主指标就是第 2 步中被量化的那个量，再加一个专门验证 X 的实验和一个专门拆掉主要质疑的实验，基线要包括加强版。
