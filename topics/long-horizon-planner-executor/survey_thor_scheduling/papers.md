# Thor 两模型调度：三个缺口的论文卡片

快照 2026-11-19。补 [brief.md](brief.md) 第 3 节列出的三个缺口：Apple 自家系统工作、Jetson Orin/Thor 上的 VLA/VLM 实测、机器人操作以外的快慢双模型系统。每个缺口精读 3 篇，共 9 篇；略读 12 篇，未收的见第 5 节。
选取方法：arXiv / 会议页检索，排除 [edge-colocation-scheduling/survey.md](../../edge-colocation-scheduling/survey.md)、[unified_memory_inference.md](../../edge-colocation-scheduling/unified_memory_inference.md)、[papers.md](../../edge-colocation-scheduling/papers.md) 已收的工作（MLX、LLM in a flash、EdgeAgent、DualVLN on Thor、OxyGen、FiS-VLA、RTC、PR #193 等）。数字均来自论文正文并注明位置；† 表示不是从论文正文核实的。

## 1. 总表

| 论文 | 缺口 | 重合度 | 一句话结论 |
|---|---|---|---|
| [AFM 2024](https://arxiv.org/abs/2407.21075) | 1 Apple | 低 | 端侧只靠减字节（约 3.7 bpw palettization + LoRA 恢复），没有多模型调度 |
| [AFM Tech Report 2025](https://arxiv.org/abs/2507.13575) | 1 Apple | 中 | KV-cache sharing 同时省 37.5 % KV 内存和 37.5 % prefill 计算，是"减少 prefill 干扰"的结构性做法 |
| [ReDrafter](https://arxiv.org/abs/2403.09919) | 1 Apple | 低 | Apple Silicon 上投机解码的最优 beam 宽度比 H100 小一个数量级，算力上限先到 |
| [EagleVLA（原 Jetson-PI）](https://arxiv.org/abs/2607.12659) | 2 Jetson | 高 | Thor/Orin 上 VLM 与 action expert 都受带宽限制，并行会争带宽，于是按置信度跳过 VLM 调用 |
| [Characterizing VLA Models](https://arxiv.org/abs/2603.02271) | 2 Jetson | 中 | MolmoAct-7B 在 Thor 上算力 5×、端到端只快 1.4×：生成阶段受带宽限制，占约 75 % |
| [Offload or Overload](https://arxiv.org/abs/2603.18284) | 2 Jetson | 中 | π0.5 在 Thor 上动作块约 144 ms；MPS 不限 SM 时共置让 π0.5 时延 +230 %，限 10 % SM 只 +3 % |
| [DriveVLM（-Dual）](https://proceedings.mlr.press/v270/tian25c.html) | 3 快慢 | 中 | 量产车上快慢两系统分放两颗 OrinX，回避共置；没有测慢模型时延对质量的影响 |
| [AsyncDriver](https://arxiv.org/abs/2406.14556) | 3 快慢 | 中 | LLM 调用间隔从 1 帧拉到 149 帧，闭环分数几乎不变：慢模型频率对质量的敏感度可以很低 |
| [Real-Time Reasoning Agents](https://arxiv.org/abs/2511.04898) | 3 快慢 | 高 | 把"环境不等推理"形式化为每步 token 预算，用分数-时间压力曲线衡量时延对质量的影响 |

## 2. 缺口 1：Apple 自家的系统工作

检索结论：Apple 发表了端侧模型技术报告和投机解码，没有发表 CPU/GPU/ANE 间带宽共享、多模型共置或带截止期调度的论文。检索词：`Apple machine learning research Neural Engine GPU concurrent scheduling memory bandwidth`、`Apple Intelligence Foundation Language Models on-device inference`、machinelearning.apple.com 站内 "on-device"、"Neural Engine"。命中的 ANE/统一内存调度工作（FusionML、ANE 架构分析、ANE batch prefill）都不是 Apple 发表的，见第 5 节。

### 1. Apple Intelligence Foundation Language Models（arXiv 预印本 2024）· [2407.21075](https://arxiv.org/abs/2407.21075)
- **一句话**：约 3B 端侧模型（AFM-on-device）与服务器模型的技术报告，含端侧推理优化。
- **intro 提出的问题 / 解决了哪些**：在端侧算力、内存、功耗约束下提供可用的通用模型和多任务能力。用低比特 + 运行时可换的 LoRA adapter 解决；与系统其他负载的资源共享只一句"管理内存并保证 OS 响应性"（§5.1），没有给机制。
- **方法：自然推导还是灵光一闪**：自然推导。端侧带宽、容量受限 → 压到 < 4 bpw → 用 LoRA 恢复精度；用 Talaria 按 op 选比特率。
- **关键数字**：ANE 上 palettization，混合 2/4 bit 平均约 3.5 bpw，产品用 3.7 bpw（§5.2）；GQA 8 个 KV head、共享输入输出 embedding（§2，Table 1）。iPhone 15 Pro 上 TTFT 约 0.6 ms/prompt token、30 tok/s（Apple 博客 2024-06-10，非报告正文）†。
- **硬件与调度画像**：统一内存（iPhone/Mac）；无截止期；单模型 + 多个 adapter 动态加载、缓存、换出（§5.1）；无调度单位；指标为质量与 latency/功耗。
- **与我们的重合度**：低。只有单模型减字节。
- **我们有、他们没考虑的**：两个模型同时争带宽；执行器硬时限；adapter 切换不是另一模型的并发访存。

### 2. Apple Intelligence Foundation Language Models: Tech Report 2025（arXiv 预印本 2025）· [2507.13575](https://arxiv.org/abs/2507.13575)
- **一句话**：第二代约 3B 端侧模型与 PT-MoE 服务器模型，端侧用 KV-cache sharing 与 2-bit QAT。
- **intro 提出的问题 / 解决了哪些**：端侧模型在 Apple silicon 上的效率与多模态、工具调用能力。解决了 KV 内存与 TTFT；端侧吞吐、并发、与其他负载共存都没有量化。
- **方法：自然推导还是灵光一闪**：KV-cache sharing 有一个非显然的观察：后 37.5 % 的层不产生 K/V、直接复用前段的 KV，则 prefill 可以整段跳过这些层，KV 内存与 prefill 计算同比例下降（§2.1）。不是顺着"压缩 KV"自然得到的，而是改结构让 prefill 变短。
- **关键数字**：Block 1 占 62.5 % 层、Block 2 占 37.5 %，KV 内存降 37.5 %，TTFT 约降 37.5 %（§2.1）；端侧 2 bpw QAT，embedding 4 bit，KV cache 8 bit（§6）；服务器 ASTC 3.56 bpw，由 GPU 固定功能硬件解压，"显著减少内存带宽"（§6）。OS daemon 用投机解码（§7），无数字。
- **硬件与调度画像**：统一内存；无截止期；单模型；无调度单位；指标为质量，系统指标只有定性陈述（§8）。
- **与我们的重合度**：中。prefill 通过算力干扰执行器，"少算 37.5 % 的 prefill"直接对应我们的干扰源；但需要重训。
- **我们有、他们没考虑的**：干扰方向（规划器 prefill 拖执行器）；batch = 1 多帧长 prompt；20 SM 下 prefill 时长对执行器尾延迟的影响。

### 3. Recurrent Drafter for Fast Speculative Decoding in Large Language Models（arXiv 预印本 2024，Apple）· [2403.09919](https://arxiv.org/abs/2403.09919)
- **一句话**：以 LLM 隐状态为条件的 RNN draft + 动态 tree attention，含 MLX 在 Apple Silicon 上的实测。
- **intro 提出的问题 / 解决了哪些**：自回归解码慢；解决了单模型解码加速。投机解码在本调研只作工具引用，收它是因为它是 Apple 唯一给出自家芯片上推理系统数字的论文。
- **方法：自然推导还是灵光一闪**：自然推导（Medusa 的 draft head 换成循环结构 + beam search 去重）。端侧部分的观察有用：最优 beam 宽度在端侧远小于服务器 GPU。
- **关键数字**：batch 1、beam length 4，M2 Ultra 上 Vicuna 33B 最高 2.28×，M1 Max 上 7B 1.32×，BW=4 时 7B/13B 反而 < 1×（§4.2，Table 2）；A100/H100 上最优 beam 宽度 ≥ 50（附录 A.2.1）。作者把 33B 归因于 memory/IO 瓶颈（§4.2）。
- **硬件与调度画像**：统一内存（M1 Max、M2 Ultra）；无截止期；单模型；调度单位为 draft-verify 迭代；指标为加速比。
- **与我们的重合度**：低。
- **我们有、他们没考虑的**：在 20 SM 的 Thor 上，verify 的额外算力会和执行器抢 SM；投机宽度应受执行器余量约束，而不只是芯片算力上限。

## 3. 缺口 2：Jetson Orin / Thor 上的 VLA/VLM 实测

### 4. EagleVLA: Towards Onboard Real-Time Robot Control via Foresight-Aligned Asynchronous Inference（CoRL 2026 Spotlight；arXiv 曾名 Jetson-PI）· [2607.12659](https://arxiv.org/abs/2607.12659)
- **一句话**：在 Orin/Thor 上让 π0.5 这类 VLM + action expert 结构异步运行，用"未来表征预测 + 置信度调度"决定每步是否重跑 VLM，并做了 llama.cpp 风格推理引擎。
- **intro 提出的问题 / 解决了哪些**：端侧 π0.5 约 1.4 s/次（Orin），异步推理带来感知-执行错位和长反应时间。两者都解决了；没有讨论硬时限、尾延迟，也没有把 VLM 与 action expert 并发执行当作调度问题，而是绕开（串行 + 跳过 VLM）。
- **方法：自然推导还是灵光一闪**：关键观察：Orin/Thor 的 balanced intensity（Thor 3736）远高于 4090（618），VLM 与 action expert 在板上都受带宽限制，并行执行会严重争带宽（§4.2，Table 2），所以减少 VLM 调用次数比并行更划算。这个判断来自 roofline 分析，论文没有给出两者并行运行的实测；结论顺着它得到。
- **关键数字**：Thor 上 π0.5 naive 分段 ViT 61.7 / LLM prefill 187.0 / action expert 209.2 / 总 457.9 ms（Table 1）；Thor 全部优化后总 132.8 ms、反应时间 61.1 ms、16.37 Hz（Table 4）；LIBERO 平均比 VLASH 高 14.8 %、比 RTC 高 3.9 %（Table 3）。
- **硬件与调度画像**：统一内存（Orin、Thor）；无截止期，真机 15 Hz 执行；VLM 与 action expert 同芯片、串行；调度单位为每个控制循环迭代（置信度 ĉ 与阈值 θ 比较，Alg. 1）；指标为反应时间、控制频率、SR。
- **与我们的重合度**：高。同芯片、同带宽结论、同"慢部分是否调用"的决策。
- **我们有、他们没考虑的**：他们的慢部分是同一模型的前段，可以跳过；我们的规划器是独立 27B 模型，必须按 ~1 Hz 调用且 L 进入任务 SR；他们用串行回避争用，我们已测得 MPS 优先级下并发 0 % 错过时限，问题变成如何分配争用。

### 5. Characterizing VLA Models: Identifying the Action Generation Bottleneck for Edge AI Architectures（arXiv 预印本 2026，Google / Purdue）· [2603.02271](https://arxiv.org/abs/2603.02271)
- **一句话**：用 Nsight 在 Orin、Thor 上分阶段剖析 MolmoAct-7B，再用 roofline 模拟器外推 GDDR7、PIM 等假想硬件。
- **intro 提出的问题 / 解决了哪些**：VLA 要扩到 10–100B 且保持 10–20 Hz，边缘加速器能否做到。回答了"瓶颈在受带宽限制的生成阶段"；没有给出解决方案，只指向软硬件协同设计。
- **方法：自然推导还是灵光一闪**：自然推导：按阶段计时 + roofline。
- **关键数字**：生成阶段约占单步时延 75 %；端侧时延比 10 Hz 所需高约 200–300×；Thor 算力是 Orin 的 5×，端到端只快 1.4×（§4.1，Fig. 2）。各阶段绝对时延只在图中†。
- **硬件与调度画像**：统一内存（Orin 203 GB/s，Thor 273 GB/s，Table 1）；无截止期；单模型；无调度；指标为分阶段时延。
- **与我们的重合度**：中。从单模型侧印证"Thor 上 decode 受带宽限制"，是我们"decode 通过带宽干扰"的旁证。
- **我们有、他们没考虑的**：第二个模型同时访存；batch = 1 的 27B NVFP4 decode；CPU 访存负载（我们测得规划器 decode 慢约 2.2×）。

### 6. Offload or Overload: A Platform Measurement Study of Mobile Robotic Manipulation Workloads（arXiv 预印本 2026，UC Berkeley / Microsoft）· [2603.18284](https://arxiv.org/abs/2603.18284)
- **一句话**：在 Orin Nano / Orin AGX / Thor、L4 / DGX Spark、A100 上测移动操作工作负载（π0.5、VLMaps、DreamZero 等）的时延、精度、功耗和共享代价。
- **intro 提出的问题 / 解决了哪些**：机器人该板载还是卸载到边缘/云，以及多机器人共享基础设施时会怎样。给出了测量；没有给调度机制。
- **方法：自然推导还是灵光一闪**：测量研究，自然推导。
- **关键数字**：π0.5 handover 动作块时延 A100 约 60 ms、Thor 约 144 ms、Orin AGX 约 440 ms，对应成功率 80 % / 80 % / 30 %（§5，Fig. 8、11）；A100 上 π0.5 与 VLMaps 共置：时间片 +75 %，MPS 不限 SM +230 %，VLMaps 限 10 % SM 时 π0.5 只 +3 % 但 VLMaps 帧时 +261 %（§7）。
- **硬件与调度画像**：板载为统一内存，但共置实验在 A100（独立显存）；无截止期；两模型同 GPU（A100）；调度单位为 MPS SM 比例 / 时间片；指标为时延增幅、成功率。
- **与我们的重合度**：中。唯一同时给出"时延 → SR"和"共置 → 时延"的测量，但两者不在同一块 Thor 上。
- **我们有、他们没考虑的**：Thor 上共置（带宽共享、CPU 也在抢）；时延 → SR 的实验是注入网络延迟（§6.1，均值 10 ms），不是规划器 L；没有尾延迟与错过率。

## 4. 缺口 3：机器人操作以外的快慢双模型系统

### 7. DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models（CoRL 2024，PMLR 270）· [2402.12289](https://arxiv.org/abs/2402.12289)
- **一句话**：VLM 做场景描述、分析、分层规划；DriveVLM-Dual 把它与传统自动驾驶栈快慢异步结合，并部署到量产车。
- **intro 提出的问题 / 解决了哪些**：长尾场景理解，VLM 空间推理弱、算力贵。Dual 结构解决了"VLM 太慢不能直接控车"；慢分支时延对规划质量的影响没有测。
- **方法：自然推导还是灵光一闪**：自然推导：VLM 出低频粗轨迹 W_slow，传统规划器细化为高频 W_fast（§3.5）。
- **关键数字**：车端两颗 OrinX，端到端系统在 OrinX-1、DriveVLM 在 OrinX-2，异步协作；DriveVLM 优化后平均 410 ms（§6）；Eagle 投机解码 decode 2.7×，0.328 s → 0.130 s（§6，Table 9）。
- **硬件与调度画像**：每颗 OrinX 是统一内存，但两模型分放两颗芯片；快分支有实时要求，未写明时限；调度单位为两个独立系统；指标为 L2、碰撞率（Table 3–4），无时延-质量曲线。
- **与我们的重合度**：中。结构与我们同构，但用双芯片回避了共置。
- **我们有、他们没考虑的**：单芯片共置；慢模型 L 对任务结果的量化；规划器 27B 而非 < 4B（他们为实时性选了 < 4B，§6）。

### 8. Asynchronous Large Language Model Enhanced Planner for Autonomous Driving（AsyncDriver，ECCV 2024）· [2406.14556](https://arxiv.org/abs/2406.14556)
- **一句话**：Llama2-13B 抽取场景-指令特征，经 Adaptive Injection Block 注入实时规划器；LLM 与规划器频率解耦，间隔期复用旧特征。
- **intro 提出的问题 / 解决了哪些**：LLM 规划器可解释但慢，实时规划器快但难控制。解决了"LLM 不必每帧运行"；两模型共享 GPU、调度、特征过期如何处理都没讨论。
- **方法：自然推导还是灵光一闪**：关键一步是让 LLM 只输出注入特征而非轨迹，于是规划器结构不变，可以按任意间隔复用旧特征。顺着"LLM 慢"自然得到。
- **关键数字**：调用间隔 [1, 9, 17, 29, 49, 79, 149] 帧，精度"基本稳定"；间隔 149 帧（每个场景一次）仍比 GameFormer 高 1.0 分以上；约 33 帧时推理时间降近 40 %、精度损失约 1 %（§4，Fig. 3；各间隔的分数只在图中†）。Hard20 闭环分数 65.00 对 GameFormer 62.05（Table 1）。推理时间在 Tesla A30 上测。
- **硬件与调度画像**：独立显存 GPU（A30）；无截止期；未说明两模型是否同 GPU；调度单位为固定帧间隔；指标为 nuPlan 闭环分数 vs 调用间隔 + 推理时间。
- **与我们的重合度**：中。给出了"慢模型频率 → 质量"的曲线形状；但是离线 nuPlan 回放，间隔是人为设定，不是由真实时延产生。
- **我们有、他们没考虑的**：L 由共置干扰决定而不是自由设定；调用间隔与决策延迟 L 是两个量（他们只扫了间隔，没有扫"输出晚到"）；任务 SR 对规划器的依赖比 nuPlan 闭环分数更强时，曲线可能不平。

### 9. Real-Time Reasoning Agents in Evolving Environments（ICLR 2026）· [2511.04898](https://arxiv.org/abs/2511.04898)
- **一句话**：Real-Time Reasoning Gym 让环境按固定节奏推进、不等推理；AgileThinker 并行运行 planning 线程（R1）与 reactive 线程（V3），reactive 读 planning 的部分推理。
- **intro 提出的问题 / 解决了哪些**：LLM agent 评测假设环境等待推理。形式化了"时间压力"并给出双线程方案；资源受限时两线程共享硬件只在附录做了一个对照。
- **方法：自然推导还是灵光一闪**：非显然的一步：用 token 数作为时间代理，T = N_T × TPOT，并验证 T = αN + β 的线性拟合 R² = 0.9986（§2、§5），使"时延对质量的影响"可以离线、可复现地扫描。
- **关键数字**：时间压力增大时 Planning（R1）分数 0.92 → 0.05，AgileThinker 0.90 → 0.58（§4）；资源共享时改为交替推理（等效 token 吞吐），Freeway 0.84 → 0.72、Overcooked 0.92 → 0.85、Snake 0.54 → 0.58，仍高于两个单范式基线（附录 C.5，Table 11）。
- **硬件与调度画像**：云 API，硬件未说明；环境步有隐式截止期（到点没有动作就执行默认动作）；主实验两模型独立吞吐，C.5 为时间片共享；调度单位为 token 预算；指标为归一化分数 vs 每步 token 预算 / 认知负载。
- **与我们的重合度**：高（指标层面）。"环境不等、到点执行默认动作、慢线程的部分输出可被快线程读"与我们的执行器-规划器同构。
- **我们有、他们没考虑的**：共享硬件只做了等吞吐时间片，没有带宽争用让两者同时变慢；快线程是 LLM 而非有硬时限的控制器；token 作时间代理在 Thor 上不成立（共置时 TPOT 随干扰变化约 2.2×）。

## 5. 综合

### 判读表各问题的答案

| 问题 | 答案 | 判读行 | 置信度 |
|---|---|---|---|
| 统一内存设备上，有没有工作把内存带宽当作调度对象，且有截止期？ | 本轮 9 篇都没有。EagleVLA 指出板上带宽争用，但选择串行 + 少调 VLM 回避；Offload or Overload 的 MPS 限 SM 实验在 A100 上做。结合前两份综述（SERENO、Agent.xpu 有带宽让出但无硬时限双 LLM），空白成立 | "没有：空白成立，记为候选问题" | 中高：检索限于 arXiv 与会议页 |
| Apple 是否发表过自家硬件上的推理系统或调度工作？ | 有推理优化（AFM 2024/2025 报告、ReDrafter），没有调度、共置或带宽共享工作。Apple 的路线是减字节（< 4 bpw、ASTC、KV 8 bit）和减 prefill（KV-cache sharing），资源共享只有一句"保证 OS 响应性" | "没有"这一行：保留 MLX、EdgeAgent 等间接证据，检索范围见第 2 节 | 中高 |
| Jetson Orin / Thor 上部署 VLA/VLM 的工作测了什么、做了什么？ | 测了单模型分阶段时延（EagleVLA Table 1、Characterizing VLA Fig. 2）、平台间 SR 与动作块时延（Offload or Overload）；做了串行异步 + 跳过 VLM、推理引擎优化。没有人在 Thor 上测两模型共置的 p99 与错过率，也没有测 CPU 访存对 GPU decode 的拖慢 | 我们的 B27/B28 测量未被做过；可对照数字：Thor 上 π0.5 naive 457.9 ms、优化后 132.8 ms（EagleVLA）、π0.5 动作块约 144 ms（Offload or Overload） | 高 |
| 快慢双模型系统用什么指标衡量慢模型的时延对质量的影响？ | 三种：质量 vs 调用间隔（AsyncDriver，闭环分数 vs 帧间隔）；质量 vs 每步时间预算（Real-Time Reasoning，分数 vs token/步）；质量 vs 反应时间（EagleVLA，SR vs Δ）。DriveVLM 没有测。都只报均值，没有 L 的分布 | 排序指标写成"SR 随 L 的曲线"，并把调用间隔与输出延迟 L 分开扫 | 中 |

### 这些论文都没有处理的问题（候选研究问题）

- **L 由共置干扰内生决定**：所有工作里慢模型的时延要么是自由参数（AsyncDriver 的间隔、Real-Time Reasoning 的 token 预算），要么靠分芯片/串行固定。没有工作把"执行器余量 → 让给规划器的带宽/SM → L → SR"作为一条链测量。
- **调用间隔与决策延迟是两个量**：AsyncDriver 只扫间隔且曲线平坦；我们的规划器背靠背调用，间隔约等于 L。缺少同时扫两者、确认 SR 对哪个敏感的实验。
- **硬时限快模型 + 软时限慢模型的不对称**：Real-Time Reasoning 的快线程超时只执行默认动作，EagleVLA 无时限。没有工作把快模型的尾延迟（p99、错过率）作为约束、慢模型 L 作为目标。
- **共享硬件下慢模型的退化方式**：Real-Time Reasoning C.5 只比较等吞吐时间片；带宽争用让两模型同时变慢、且 prefill 与 decode 的干扰通道不同（算力对带宽），没人测过。
- **减 prefill 计算作为调度手段**：AFM 2025 的 KV-cache sharing 说明结构改动能同比例减少 prefill；在 Thor 上，减少规划器 prefill 的 SM 占用时间是否直接降低执行器 p99，没有数据。

## 6. 考虑过但未收

- [LiteVLM](https://arxiv.org/abs/2506.07416)（NVIDIA，DRIVE Thor）：InternVL2.5 2B 分段时延 529.7 → 213.6 ms（Table 1），单模型加速，可作为 Thor 上 VLM 分段数字的对照；与缺口 2 的三篇相比没有 VLA 或调度内容。
- [Cross-Platform Scaling of VLA Models](https://arxiv.org/abs/2509.11480)：只有 Orin，且时延与吞吐数字互相不一致（0.03 ms 对 474.78 Hz）。
- [Accelerating VLA Inference with Perturbation Informed Execution（PICES）](https://arxiv.org/abs/2610.19656)：Orin/Thor 上 2.3× 墙钟加速来自减少调用次数，属于执行器侧算法，与共置无关。
- [Don't Run with Scissors](https://arxiv.org/abs/2510.08464)：Thor / Orin NX 上 VLA 剪枝时延，单模型压缩。
- FusionML（[2607.22785](https://www.alphaxiv.org/abs/2607.22785)）、Apple Neural Engine 架构分析（[2606.22283](https://arxiv.org/abs/2606.22283)）、ANE batch prefill（[atomgradient](https://atomgradient.github.io/hybird-batch-prefill-on-ane)）：Apple Silicon 上 CPU/GPU/ANE 共享带宽的第三方工作，不是 Apple 发表；未读全文†。
- Apple 博客 "Introducing Apple's On-Device and Server Foundation Models"（[链接](https://machinelearning.apple.com/research/introducing-apple-foundation-models)）：只作为 AFM 2024 端侧时延数字的出处，非论文。
- FASIONAD、Senna、LeapVAD 等自动驾驶快慢系统：与 DriveVLM-Dual、AsyncDriver 结构相近；本轮未打开核实是否含时延-质量测量†。
- SwiftSage 等快慢 LLM agent：没有时间压力设定，时延不影响质量†（未重读原文）。
