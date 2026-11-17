# OxyGen：把同一 MoT VLA 内"每帧动作 + 跨帧语言"两路任务的 KV cache 统一管理，共享 prefill、跨帧合批 decode

- **OxyGen** · 2026 · NeurIPS 2026（主会，poster）· [arXiv 2603.14371](https://arxiv.org/abs/2603.14371)，[neurips.cc poster 149556](https://neurips.cc/virtual/2026/poster/149556)
- OxyGen: Unified KV Cache Management for VLA Inference under Multi-Task Parallelism
- 作者：Xiangyu Li, Huaizhi Tang, Xin Ding, Weijun Wang, Ting Cao（通讯）, Yunxin Liu（清华 AIR / 清华电子系 / 中科大）；精读版本 arXiv v5（2026-10-16；PDF 页眉日期 2026-10-13）。版本史：v1 2026-03-15，v2 2026-05-18，v3 2026-10-13，v4 2026-10-14，v5 2026-10-16。代码 github.com/air-embodied-brain/OxyGen。

### 会议核实

- arXiv abs 页 Comments 字段："NeurIPS 2026"。
- v5 PDF 第 1 页脚注："Accepted at the 40th Conference on Neural Information Processing Systems (NeurIPS 2026)."——这是主会论文的标准措辞；workshop 论文通常写 workshop 名。
- neurips.cc/virtual/2026/poster/149556：页面标题 "NeurIPS Poster OxyGen: …"，展示时间 Tue, Dec 8, 2026, 10:00 AM – 1:00 PM AEDT，位置 "Hall 1-4 #S2403"（悉尼会场）。同一调研中 workshop 论文（DualVLN on Thor）的 ID 是 170076，属另一编号段。
- 结论：NeurIPS 2026 主会 poster。页面没有显式写 "Main Track" 还是 "Datasets & Benchmarks Track"，因此"主会但具体 track"这一点 † 未逐字核实；OpenReview 检索 API 未返回该论文，未核对 OpenReview 页。

## 论文做了什么

MoT VLA（π0.5 这类：共享 VLM backbone + 动作专家 + 语言专家）在机器人上要同时出每帧的动作块和跨多帧的语言（记忆、旁白、QA）。现有做法（openpi 等）对两路任务各跑一次 forward（"isolated execution"），重复 prefill 同一观测，并且两路互相拖慢。OxyGen 把 KV cache 当作跨任务、跨帧的共享资源：每帧只 prefill 一次，动作专家和新语言请求共用；所有未完成的语言请求跨帧组成一个 batch，每帧前进固定 k 个 token。在 RTX 4090 和 Jetson AGX Thor 上报告动作频率与语言吞吐同时最高提升 3.7×（4090）/ 3.3×（Thor，仅图），并在板载 Thor 的人形机器人上测了分段延迟。

## 结构

- §1 Introduction：multi-task parallelism 场景；isolated KV 是根因；两条 challenge。
- §2 Related Work：MoT VLA；VLA 推理系统（模型级 / 执行级）；LLM serving 中的 KV 管理。
- §3.1 Preliminaries and Problem Formulation：式 (1)(2) prefill 与两专家；式 (3) 优化目标。
- §3.2 Unified KV Cache Manager：可恢复生成状态 σ_t（式 4）；CRUD 接口；每帧两阶段流程；k 的作用；扩展讨论（更多专家、投机解码）。
- §4.1 Setup：π0.5；LIBERO / DROID / ALOHA 三种输入配置；4090 与 Thor；两个基线；指标。
- §4.2 End-to-End Results：图 4（频率–吞吐 Pareto）、图 5（加速比热图）。
- §4.3 Ablation：图 6（KV 共享 / 合批 / MPS / 上界，4090 与 Thor）。
- §4.4 跨骨干与多专家：表 1（Xiaomi-Robotics-0、StarVLA，2–4 个专家，Qwen3.5 0.8B–9B）。
- §4.5 不同负载：均匀到达、Poisson 到达、长短混合（图 7）。
- §4.6 动作质量与资源：LIBERO 成功率；内存、功耗、能耗。
- §4.7 Real-Robot Deployment：板载 Thor 人形机器人分段延迟（图 8）。
- §5 Conclusion / Limitations。
- 附录 A：生成公式、manager 接口、算法 1。B：π0.5 补充（ALOHA/DROID、N 与 k 扫描、成功率表 2、内存能耗表 3）。C：跨模型（表 4、表 5、张量并行）。D：运行时分析（D.1 prefill 复用分段表 6；D.2 KV 管理开销表 7；D.3 3000 帧稳定性图 12；D.4 语言完成时间）。E：文本记忆应用（LoRA）。

## 问题表述

- 场景：单 GPU 设备（RTX 4090 或 Jetson AGX Thor），一个 π0.5 模型；每帧一个新观测（图像 + 指令）、一个新语言请求；动作 H = 10 步、S = 10 去噪步（§4.1.3），语言每请求最多 N 个 token。动作有"每帧截止期"以维持目标动作频率 f_min，语言是跨帧的软截止期（§3.1）。没有给出具体的截止期数值。
- 优化目标（§3.1 式 3，v5 原文）：稳态下每帧产出 H 个动作并让 B 个语言请求各前进 k 步，帧延迟 T，则 f = H/T，τ = Bk/T，

  max τ = Bk/T  s.t.  f = H/T ≥ f_min。

  原文补充："action frequency counts generated actions; the replanning rate is 1/T"。
- f_min 怎么选：v5 正文没有给出 f_min 的取值，也没有任何实验按某个 f_min 去选 k 或判定满足与否。v2（§3.1）曾写 "sustaining a minimum control frequency f_min for smooth robot control (e.g., 50Hz for dexterous manipulation)"，并称 k 是 "calibrated per-frame step budget so action meets its hard deadline"；v5 删掉了 50 Hz 例子和 "hard deadline" 措辞，改成 "our current implementation uses a fixed k. In deployment, k could be chosen to meet action-frequency and language-response requirements by profiling the expected workload on the target hardware"（§3.2）。即：式 (3) 是动机性的表述，实验中只扫描 k ∈ {1, 5, 10}、N ∈ {5,…,30}，报告结果点，约束本身没有被求解或检验。
- 指标（§4.1.3）：action frequency = H / 平均端到端帧延迟（均值）；语言吞吐（tok/s，所有活跃请求合计）；平均 batch size；峰值内存（GiB）；平均功率（W）；每请求能耗（J）。另有附录 D.4 的语言请求完成时间（中位数）。

## 核心观察

1. 重复 prefill 是显著成本：π0.5-LIBERO、4090、S = 10 时一次 prefill 约 44.0 ms，动作去噪约 20.5 ms，单步 decode 约 13 ms（表 6，D.1）；只做 KV 共享，帧延迟 200.3 → 145.1 ms（N = 5，1.4×，§4.3）；节省的绝对量约 47–48 ms（表 6）或 55–58 ms（§4.3 文字），相对比例从 N = 1 的 36.4% 降到 N = 30 的 9.4%（表 6）。证据：分段计时，"Ours w/o Batching" 与 Baseline 对比。
2. 语言越长，decode 越主导帧延迟，跨帧合批收益越大：KV 共享已开时，N = 30、k = 5 合批让帧延迟 469.3 → 175.8 ms，动作频率 21.3 → 56.9 Hz（2.7×，§4.3，4090）。合批后动作频率随 N 增长几乎不变，4090 约 60 Hz、Thor 约 27 Hz（§4.3 文字；图 6）。
3. decode 的 batch 增大几乎不加时间：Qwen3.5 骨干下 batch 1 → 30，单 token decode 4B 从 16.0 → 18.2 ms，9B 从 20.7 → 27.2 ms（附录 C，4090）。
4. MPS 多进程并行在 Thor 上比顺序执行还慢（§4.2 观察 iii："MPS parallelization offers limited gains on RTX 4090 and reduces performance on Jetson AGX Thor"）。作者给出的原因："the two processes still repeat prefix computation and compete for GPU resources"。正文没有给出 Thor 上 Parallel 的数字，只在图中：
   - 图 6 右（Thor，LIBERO，k = 5，仅图、目测）：N = 5 时 Parallel 约 19 Hz、Baseline 约 22.5 Hz；N = 30 时 Parallel 约 7 Hz、Baseline 约 9.5 Hz。4090（图 6 左）则相反，N = 5 时 Parallel 约 55 Hz 高于 Baseline 约 50 Hz。
   - 图 4 右（Thor，仅图、目测）：Parallel 各点（约 7.5–19 Hz、吞吐约 9–21 tok/s）都在 Baseline 各点（约 10–22 Hz、约 11–29 tok/s）的左下方。
   - v2（§4.6 / 表 2 对应文字）对 4090 上 Parallel 能耗 +3% 的解释是 "redundant computation and memory bandwidth contention"；v5 删去了 "memory bandwidth contention"，只保留 "nearly doubles peak memory and increases energy per request by 3%"（§4.6，表 3：峰值内存 6.43 → 12.49 GiB，功率 293.5 → 324.4 W，能耗 117.4 → 120.9 J/req）。
   - MPS 的配置（是否限 SM 百分比、客户端优先级、两进程怎样同步到同一帧）论文没有写。
5. KV 管理本身有开销：batch 30 时 manager 占帧延迟 17.5%（JAX）/ 6.6%（PyTorch），主要是 cache 拆分（split 18.24 ms / 8.53 ms），但总帧延迟仍下降（147.6 → 120.6 ms JAX；168.3 → 148.1 ms PyTorch）（表 7，D.2，4090）。
6. 负载变化下动作频率较稳：长短混合请求时动作频率约 59–62 Hz，平均 batch 1.1–3.5（§4.5，图 7，4090）；均匀 / Poisson 到达率增加时动作频率下降（仅图）。

## challenge

- 异构专家的 KV 语义：同一个 prefill 结果要同时服务"去噪时反复读前缀"的动作专家和"跨帧不断扩展自己 cache"的语言专家，生命周期不同（§1 challenge i，§2 末段）。
- 帧边界内的语言解码：标准 continuous batching 不和周期性的动作推理协调；控制回路要求语言解码在每帧预算内前进、保存状态、下一帧不重算地继续（§1 challenge ii）。

## 机制

- 统一 KV cache manager M（§3.2，附录 A.2）：每个语言请求一个可恢复状态 σ_t = (K_t, y_t, δ_t)（式 4），提供 Store / Retrieve / Update / Remove 和 Batch / UnBatch；请求 cache 在开始时预留 N 个 token 的容量直到完成（A.2 "Request storage and allocation"）；只把"前缀与 cache 容量一致"的状态合批，不同请求保留各自位置与 attention mask。
- 跨任务 KV 共享：每帧 prefill 一次，K_t 同时给动作专家（只读，去噪 S 步）和新语言请求的初始状态（§3.2）。
- 跨帧 continuous batching：所有未结束请求堆成 σ̂，做 k 步批量自回归 decode，结束的请求移出（§3.2，算法 1）。
- 依赖：纯框架层改动，在 openpi 上实现，JAX 与 PyTorch 两个后端（§1，D.2）；不依赖 MPS、green context、stream 优先级或任何驱动能力。Thor 上的 JetPack、精度、功耗模式未写。

### 算法 1 的每帧调度（问题 4）

- 每帧顺序执行，不是时间片，也不是在一个 forward 内交错：
  1. 第 1 行 Prefill(o_t) → K_t；
  2. 第 2 行 ActionDenoise(K_t, S) → A_t（动作在这里就产出）；
  3. 第 3–4 行用 K_t 初始化新语言状态并 Store；
  4. 第 5–9 行把新请求并入活跃集合，Retrieve → Batch → BatchedLanguageDecode(σ̂, k) → UnBatch；
  5. 第 10–15 行逐请求取出新 token，结束的 Remove，其余 Update。
- 动作与语言之间是串行的两段：动作阶段（prefill + 去噪）在前，语言阶段（k 步合批 decode）在后；帧延迟 T = prefill + 去噪 + k 步合批 decode + 管理开销。动作阶段内没有被语言打断的情况，所以论文里不存在"语言抢占动作"的问题；代价是下一帧的动作必须等本帧 k 步 decode 结束。
- 人形机器人实验（§4.7）把这一点用作延迟隐藏：动作发出后，195 ms 的语言 decode 可以与 333 ms 的物理执行重叠。论文写的是 "can be hidden"，没有给出重叠后的实测帧周期。
- k 由什么决定：实现中是固定常数，实验扫描 k ∈ {1, 5, 10}（§3.2，§4.2）；没有自适应、没有按截止期在线调整。稳态 batch B = N/k（§4.2），单个请求完成需 ⌈N/k⌉ 帧，约 ⌈N/k⌉·T（D.4；4090，N = 30，k = 5，T = 175.8 ms，估计 1.055 s，实测中位 1.057 s）。k 越小，batch 越大、帧越短、动作频率和总吞吐越高，但单请求语言完成时间越长（§3.2，D.4）。

## 评估

- 基线（§4.1.2）：
  - Baseline（顺序 isolated execution）：同一帧内动作与语言各自独立 forward，顺序执行；openpi 未开源语言生成，按社区复现实现（脚注 1）。
  - Parallel（并行 isolated execution）：动作与语言放在同一 GPU 的两个进程里，经 CUDA MPS 并发。
  - 消融："Ours w/o Batching"（只开 KV 共享）；"Batching Upper Bound"（每帧只 decode k 个 token、丢掉剩下 N − k 个，作为上界 oracle，§4.3）。
- 负载：π0.5，LIBERO / DROID / ALOHA 三种输入配置；每帧 1 个新请求；N ∈ {5, 10, 15, 20, 30}，k ∈ {1, 5, 10}，S ∈ {5, 10, 15, 20}（图 5、图 10）；另有均匀 / Poisson / 长短混合到达（§4.5，仅 4090）；3000 帧波动负载（D.3，仅 4090）。
- 主要结果：
  - 4090：最高 3.7×；同时 >200 tok/s 与 70 Hz（摘要；图 1 标注动作频率 19.1 → 70.5 Hz、语言吞吐 57.2 → 212.9 tok/s）；相同输出长度下两项提升 1.2–3.7×（§4.2）。
  - N = 30 时动作频率加速比 2.10–3.74×，主要由 k 决定（图 5 热图数字）；k = 5 时随 N 从 5 到 30 加速比 1.20–3.05×（图 10）。
  - 跨骨干（表 1 / 表 4，4090，N = 30）：Xiaomi-Robotics-0 k = 1 5.83×、k = 5 3.23×；StarVLA Qwen3VL-PI_v3 6.64× / 3.60×；加 GR00T（3 专家）6.05× / 3.53×；再加 OFT（4 专家）6.30× / 3.69×。N = k = 10 时 Xiaomi-Robotics-0 为 0.96×（比基线慢）（附录 C）。
  - Qwen3.5 0.8B–9B：N = 30、k = 5 加速 3.58–3.79×，基线帧延迟 429.0–731.9 ms（表 5）。
  - 张量并行（9B，两块 4090 经 PCIe）：TP=1 3.50×，TP=2 3.71×，帧延迟 493.4 → 535.3 ms（附录 C）。
  - 成功率：LIBERO 10 seed、20,000 rollout，96.76%（95% CI 96.55–96.97%）对 openpi 公布的 96.85%（§4.6，表 2）。
  - 资源（4090，表 3）：峰值内存 +15%（6.43 → 7.41 GiB），平均功率最多 −47%（293.5 → 154.4 W），每请求能耗最多 −78%（117.4 → 25.8 J）。
  - 稳定性（D.3，4090，k = 4，3000 帧，最多 40 个活跃请求）：活跃内存 6.57–6.94 GiB（JAX）、7.29–7.65 GiB（PyTorch）；每 119 帧周期平均帧延迟首尾五个周期 245.6 → 249.8 ms（JAX）、315.5 → 320.9 ms（PyTorch）；GPU 温度 87–90 °C。

### Thor 上的全部数字（问题 2）

- §4.3 文字：合批后动作频率维持约 27 Hz（LIBERO，k = 5，N 增大时）。
- 图 4 右（仅图）：Thor 上标注 3.3×；Ours（k = 1）最高约 31 Hz、约 95 tok/s（目测）。
- 图 9（附录 B.1，ALOHA / DROID）：Thor 上同样标 3.3×（仅图）。
- 图 6 右（仅图，目测）：Ours 约 25–31 Hz；Ours w/o Batching 约 10.5–30.5 Hz；Baseline 约 9.5–22.5 Hz；Parallel 约 7–19 Hz；上界约 30.5 Hz。
- MPS Parallel：§4.2 只说 "reduces performance on Jetson AGX Thor"，原因见核心观察 4；具体降幅正文无数字，仅图（约比 Baseline 低 2.5–3.5 Hz，即约 15–25%，目测）。
- 人形机器人板载 Thor（§4.7，图 8 表）：三路 RGB 224×224，H = 10，S = 10，N = 30，k = 5。

  | 阶段 | Baseline (ms) | OxyGen (ms) |
  |---|---|---|
  | Video stream | 9.4 | - |
  | Action path（prefill + 去噪） | 207.5 | 198.0 |
  | Language path | 822.3 | 195.4 |
  | Action comm. | 0.6 | - |
  | Action exec. | 333.0 | - |

  合计 1030 ms → 393 ms。Baseline 的语言路径含第二次 prefill + 30 步 decode；OxyGen 的语言路径是复用 prefill 后的合批 decode（图 8 说明）。论文没有给出机器人上的动作频率，也没有给出重叠后实际控制周期。若按 f = H/T 计算，393 ms 对应约 25 Hz（判断：自行换算，非论文数字），与 §4.3 的约 27 Hz 量级一致。
- Thor 的软件栈（JetPack、CUDA、精度、功耗模式、是否用 CUDA Graph）未写。

### 尾延迟 / 抖动 / 截止期错过（问题 3）

- 全文（v5 正文 + 附录 A–E）和 v2 中都没有 tail latency（p95 / p99）、jitter、deadline miss 率或最坏帧延迟。
- 所有频率指标都基于平均帧延迟（§4.1.3）。唯一的分布性统计是 D.4 的语言请求完成时间中位数 1.057 s，以及 D.3 的"每 119 帧周期平均帧延迟"和内存 min–max。D.3 只说 "modest average-latency drift"，并提到 87–90 °C 温度 "can affect sustained latency"，没有分布。

- 消融：图 6（KV 共享 vs 合批 vs MPS vs 上界，两平台）；表 6（prefill 复用逐 N 分段）；表 7（manager 开销）；表 4（专家数 2→4 时 KV 共享每帧节省约 25 / 50 / 75 ms）。

## 作者自述的局限

- §5 Limitations：只评估了动作 + 语言并发；需要在带视频等更多输出模态的统一模型上验证；需要与异步控制和模型级优化集成；需要端到端的真实应用（数据采集、多任务训练、闭环部署）。
- §3.2：当前实现用固定 k；部署时 k 应通过在目标硬件上 profiling 来选。
- D.2：batch 30 时 KV 拆分开销显著，"Persistent cache slots could reduce the cost"。
- D.3：温度可能影响持续延迟。
- 附录 C：N = k 时没有跨帧合批机会，Xiaomi-Robotics-0 在 N = k = 10 时 0.96×。

## 与本场景的前提对照

| 前提 | 论文 | 本场景（见 [survey.md](../survey.md) §0.2） | 是否成立 |
|---|---|---|---|
| 硬件 | RTX 4090；Jetson AGX Thor（含板载人形机器人） | Jetson AGX Thor | 部分成立（Thor 有，但软件栈与功耗模式未写） |
| 模型数 | 一个 MoT VLA（π0.5），共享 VLM backbone | 两个独立模型：ImageWAM 执行器 + Qwen3.8-27B 规划器 | 不成立 |
| 输入是否共享 | 两路任务共享同一帧观测与同一 prefill | 两模型输入、权重、tokenizer 各不相同 | 不成立 |
| 语言负载规模 | prefill 约 44 ms（4090），每请求 N ≤ 30 token，每帧 1 个新请求 | 4–6k token prefill 1.2–2.2 s，回复 14–80 token，每 5.33 s 一次 | 不成立（论文的瓶颈是 decode，本场景的主要干扰源是 prefill） |
| 调度粒度 | 每帧串行：动作阶段后接 k 步合批 decode；同一进程、同一 forward 序列 | 两模型并发，执行器周期 200 ms，规划器 prefill 以 2048 token 块（约 1.17 s）为不可分单位 | 不成立（本场景无法把规划器 prefill 塞进执行器帧内） |
| 截止期 | 名义上动作有每帧截止期、f_min；实验未设定 f_min，也未测错过 | 执行器严格 200 ms 截止期，关注 p99 和错过率 | 部分成立（目标形式相同；度量不同） |
| 指标 | 平均帧延迟换算的频率、吞吐、能耗 | p50 / p99、错过率、规划器 p95 L | 不成立 |
| MPS 行为 | Thor 上 MPS 两进程比顺序执行慢（仅图，约 15–25%）；配置未写 | MPS 客户端优先级（不限 SM）是目前最好的手段 | 不矛盾：论文的 Parallel 两进程都重复 prefill 且无优先级（判断：未写配置，推测为默认 MPS） |
| 内存结构 | 4090 独立显存；Thor 统一内存未讨论 | 128 GB 统一内存，CPU/GPU 共享约 273 GB/s | 论文未涉及带宽干扰（v5 删去了 v2 的 "memory bandwidth contention" 解释） |

## 可以借用的 / 不能借用的（问题 5，均为判断）

- 判断，可借用：
  - 式 (3) 的问题形式——"在执行器频率 / 截止期约束下最大化另一方吞吐或时延"——可以直接用来表述本场景，但需要把约束从均值频率换成 p99 / 错过率，并把目标从 token 吞吐换成规划器延迟 L。
  - "动作阶段在前、非关键工作在后"的帧内顺序，以及"动作发出后把语言工作藏进物理执行窗口"（§4.7）的思路，可迁移为：在执行器每次调用完成到下一次调用之间的空隙放行规划器的块。前提是规划器工作能切到小于空隙的粒度；本场景规划器 prefill 块约 1.17 s，远大于 200 ms 周期 − 37.5 ms 的空隙，所以只对规划器 decode（约 27 ms/token）有直接意义：每个执行器周期内让规划器 decode 前进 k 个 token，就是算法 1 的 k 在两模型下的对应物。
  - k 作为"每帧给非关键任务的预算"这一控制变量，以及 ⌈N/k⌉·T 的完成时间估计（D.4），可以用来估规划器 decode 段的延迟代价。
  - 论文在 Thor 上观察到"两进程 MPS 并行比顺序还慢"，可以作为一个外部数据点：在 Thor 上不加控制的进程级并发可能是负收益。但论文的 Parallel 同时包含重复 prefill，不能把慢的部分全部归因于并发干扰。
- 判断，不能借用：
  - 跨任务 KV 共享：两个模型权重和表征不同，没有可共享的 prefill。论文提到的 cross-model KV transfer（Cache-to-Cache、DroidSpeak，§3.2 讨论）需要模型间对齐，不在本场景范围。
  - 跨帧 continuous batching：本场景规划器每 5.33 s 只有一个请求，没有多个并发语言请求可合批，batch 恒为 1。
  - "同一进程串行执行"的隔离方式：本场景要求执行器 37.5 ms 调用不被 1.17 s 的规划器 prefill 块挡住，串行顺序做不到，必须靠并发 + 优先级 / SM 划分，这是论文没有处理的问题。
  - 论文的所有收益都来自去掉重复计算与合批摊薄权重读取，不是来自干扰控制；本场景的核心问题（prefill 抢 SM、decode 抢带宽、CPU 内存负载）在论文中没有对应测量。

## 待核实

- neurips.cc 页面未显示 track 名（Main Track vs. Datasets & Benchmarks）；OpenReview 页面未找到。
- Thor 上 Parallel 与 Baseline 的具体数值（只在图中）；可从项目仓库或作者的原始数据确认。
- Thor 上的 MPS 配置（SM 百分比、客户端优先级）、JetPack / CUDA 版本、精度（bf16 推测，未写）、功耗模式。
- 人形机器人上的实际控制周期与语言是否真的与执行重叠（论文只给分段和），以及该平台的机器人型号。
- 4090 上 KV 共享节省量：§4.3 写 55–58 ms，表 6 推出约 47–48 ms，两处测量条件（N 范围、是否同一次运行）未说明。
