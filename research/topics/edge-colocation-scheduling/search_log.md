# edge-colocation-scheduling 检索记录

- 快照日期：2026-11-17
- 每类记录检索词、来源、元数据核对方式，以及候选清单和取舍理由。
- 下列 5 篇写过略读卡片，但为把目录控制在 30 篇以内，没有收入 [papers.md](papers.md)：
  - FastServe（arXiv 2305.05920，预印本）：同一模型内 token 级抢占与 MLFQ，属于请求调度，与两模型共置的距离比 B 类其他条目远；
  - POD-Attention（ASPLOS '25）：在一个 kernel 内按 SMID 分配 prefill / decode 的 CTA，只适用于同一模型的两个阶段；
  - Helix（Figure AI 技术博客，2025-02-20）：两块独立 GPU 各跑一个系统，没有共置干扰，也没有可核对的论文条目；
  - GR00T N1（arXiv 2503.14734）：只给出 System 2 / System 1 的频率与 L40 时延，部署方式未写；
  - Mascareñas González et al., iGPU Memory Interference vs. L2 Cache Locking（WCET 2023）：Orin 上 CPU 核对 iGPU 的 LPDDR 干扰约占 WCET 的 25 %，机制是 L2 锁定，与本场景距离较远。

## A. GPU 共享与抢占

#### 检索词（WebSearch）

- `REEF microsecond-scale preemption GPU DNN inference OSDI 2022`
- `Orion interference-aware fine-grained GPU sharing ML applications EuroSys 2024 arxiv`
- `XSched preemptive scheduling diverse XPUs OSDI 2025`
- `LithOS operating system efficient machine learning GPUs SOSP 2025 arxiv`；`"LithOS" SOSP '25 doi 10.1145 Coppock`
- `GCAPS GPU context-aware preemptive priority-based scheduling real-time ECRTS`
- `TGS transparent GPU sharing container deep learning training NSDI 2023`
- `Paella low-latency model serving virtualized GPU scheduling SOSP 2023`
- `BLESS bubble-less spatial-temporal GPU sharing EuroSys 2025`
- `Usher holistic interference avoidance resource optimized ML inference OSDI 2024`
- `LLM co-location latency-critical inference best-effort same GPU preemption 2025 arxiv`
- `Jetson Orin real-time GPU scheduling multiple DNN co-execution RTSS RTAS 2024 2025 deadline`

#### 来源与元数据核对方式

- 正文：usenix.org 论文 PDF（REEF、Clockwork、XSched 来自 IPADS 主页的 OSDI'25 PDF）、ACM 版 PDF 镜像（Orion、Paella、BLESS 作者主页）、CMU PDL 的 SOSP'25 版 PDF（LithOS）、arXiv HTML/PDF（GCAPS v1、Hummingbird v2）。PDF 均用 pdftotext 抽取后阅读摘要、引言、动机与评测节。
- 元数据：以论文 PDF 首页/页脚的会议印记与 DOI、usenix 会议页面、Dagstuhl LIPIcs 页面（GCAPS）、arXiv abs 页面（Hummingbird）核对。DBLP 被反爬页面拦截，未能用于核对。
- 前向/后向引用：从 Orion、LithOS、Hummingbird、BLESS 的相关工作与基线列表（REEF、Orion、TGS、Paella、GSLICE、Zico、Tick-Tock、MIG、libsmctrl、Green Context）扩展；从 XSched 引用 [123] 追到 GCAPS 作者组的 Tegra TSG 抢占工作。

#### 候选清单（一行一篇）

| 论文 | 状态 | 理由 |
|---|---|---|
| REEF (OSDI'22) | 收录 | 种子；RT/BE 双类 DNN 推理、kernel 级抢占的代表 |
| Orion (EuroSys'24) | 收录 | 种子；干扰感知（compute vs memory bound）空间共享的代表 |
| Clockwork (OSDI'20) | 收录 | 种子；"不并发换可预测性"的代表立场 |
| Paella (SOSP'23) | 收录 | 种子；绕过硬件调度器、软件决定 kernel 顺序 |
| XSched (OSDI'25) | 收录 | 跨 XPU 的抢占抽象；含 Jetson Orin SoC 上 GPU/NPU 共享带宽的实验 |
| LithOS (SOSP'25) | 收录 | TPC 粒度空间调度 + kernel 切分；与 MPS/MIG/TGS/REEF/Orion 全面对比，含 LLM |
| GCAPS (ECRTS'24) | 收录 | 实时系统方向；Jetson 上驱动级 TSG runlist 抢占与可调度性分析 |
| BLESS (EuroSys'25) | 收录 | 用多个预建 SM 受限 MPS context 做时空共享，机制与本场景的 MPS 百分比最接近 |
| Hummingbird (预印本 2026) | 收录 | LLM 推理（高优先级）+ LLM 离线推理/训练（低优先级）；直接测了 Orion/REEF 在 LLM decode 下的带宽干扰 |
| TGS (NSDI'23) | 排除 | 容器云中训练作业的 OS 层速率控制，目标是吞吐与生产/机会作业隔离；无截止期；LithOS 已把它作为基线测过（LithOS §8.1） |
| USHER (OSDI'24) | 排除 | 集群级放置/批大小/副本数的装箱调度，不是单 GPU 内的抢占或时序控制 |
| GSLICE (SoCC'20)† | 排除 | 未核实会议元数据；MPS 空间划分 + 自适应调整，BLESS 已将其作为基线（BLESS §6 基线列表） |
| Gpulet (ATC'22)† | 排除 | 未核实会议元数据；多模型推理的空间划分与集群调度，偏吞吐 |
| Tally (arXiv 2410.07381)† | 排除 | 会议未核实；与 Orion/REEF 同族（非侵入隔离），预算有限 |
| libsmctrl / "Hardware Compute Partitioning on NVIDIA GPUs"（RTAS'23）† | 排除 | 会议未核实；TPC mask 逆向，LithOS 的 TPC 调度基于它（LithOS §7）；与 green context 同层机制 |
| ConServe (arXiv 2410.01228；作者页标 ICML'26)† | 排除 | 同一 LLM 引擎内在线/离线请求共服务，属于 LLM serving 类，不是两个独立模型共享 GPU |
| Efficient LLM Serving on Hybrid Real-time and Best-effort Requests (arXiv 2504.09590)† | 排除 | 同上，LLM 引擎内部请求调度 |
| Topology-aware Preemptive Scheduling for Co-located LLM Workloads (arXiv 2411.11560) | 排除 | 集群级拓扑感知，不是单 GPU |
| curtSCHED (RTNS'25)† / SGPRS (arXiv 2406.09425) / RED (arXiv 2308.15368) / D-STACK (arXiv 2304.13541) | 排除 | 嵌入式/实时多 DNN 调度，模型均为小 CNN，无 LLM；预算有限，仅记录 |

† = 会议元数据未在会议论文集页面/arXiv/DBLP 上核对。

---

## B. LLM 服务的阶段调度

#### 检索词
- arXiv API 按 ID 批量取元数据：2401.09670, 2403.02310, 2311.18677, 2410.18038, 2504.19867, 2305.05920, 2406.03243, 2401.11181
- "Nexus proactive intra-GPU disaggregation prefill decode LLM serving arXiv"
- "DuetServe prefill decode spatial multiplexing SM partitioning arXiv"
- "Drift LLM serving prefill decode PD multiplexing green context single GPU arXiv"
- "Splitwise ISCA 2024 ieee phase splitting Patel"
- "semi-PD Hong phase-wise disaggregated computation unified storage accepted conference"
- "Bullet spatial-temporal orchestration prefill decode SM partition LLM serving arXiv 2025"
- "Drift PD-Multiplexing Cui Chen accepted venue 2025 2026"
- "LLM inference co-located with real-time periodic deadline task GPU SM partition robot edge prefill decode interference"

#### 来源
- arXiv abs / HTML 全文（Drift、Nexus、DuetServe、semi-PD、POD-Attention、FastServe、Splitwise、Bullet）
- USENIX OSDI '24 论文页与 PDF（DistServe、Sarathi-Serve；PDF 经 pdftotext 读正文）
- PMLR v306 论文页（DuetServe）
- 作者主页 camera-ready PDF 首页 ACM 引用格式（Bullet）
- DBLP API 返回 HTTP 429，未能使用；会议名以 arXiv comment、USENIX 页面、PMLR 页面或 camera-ready 首页为准

#### 候选清单（一行一篇）
| 论文 | 处理 | 理由 |
|---|---|---|
| DistServe（OSDI '24） | 入选 | 种子；跨 GPU 分离的代表，goodput 定义的来源 |
| Sarathi-Serve（OSDI '24） | 入选 | 种子；chunked prefill + token budget 约束 TBT，与本场景"2048 块 prefill"直接相关 |
| FastServe（预印本） | 入选 | token 级抢占 + MLFQ，回答"可抢占的优先级调度"问题 |
| POD-Attention（ASPLOS '25） | 入选 | 单 kernel 内按 SM 分配 prefill / decode CTA，论证 stream / MPS 不保证 SM 级共置 |
| semi-PD（预印本） | 入选 | 用 MPS 百分比划分 SM 给 prefill / decode 两个进程，最接近本场景的 MPS 形态 |
| Drift / PD-Multiplexing（预印本） | 入选 | 用 green context 预建 SM 分组，有并发带宽争用的大样本测量 |
| Nexus（预印本） | 入选 | green context；显式测量 decode 受 prefill 的带宽争用（36 %） |
| Bullet（ASPLOS '26 †） | 入选 | MPS + libsmctrl SM mask，含争用项的性能模型，微秒级重配置 |
| DuetServe（ICML 2026） | 入选 | libsmctrl TPC 粒度划分，只在预测 TBT 违约时开启分区 |
| Splitwise（ISCA '24 †） | 排除（已略读） | 跨机器分离 + 异构 GPU / 功率，前提与单 SoC 相差最远；SLO 用相对无争用的 slowdown 倍数（Table VI：TTFT P50/P90/P99 = 2×/3×/6×，TBT 1.25×/1.5×/5×），这一定义方式记录在此 |
| TetriInfer（arXiv 2401.11181，预印本） | 排除 | 跨实例分离，机制与 DistServe 重叠，预算内优先 intra-GPU 工作 |
| Llumnix（OSDI '24，据 arXiv comment） | 排除 | 跨实例迁移做负载均衡与优先级，集群级，单设备不适用 |
| NanoFlow（OSDI '25，未核实） | 排除 | 单设备内 operator 级重叠以提吞吐，非 SLO 导向；Bullet 已作为基线覆盖 |
| MuxServe（多 LLM 空分时分复用） | 排除 | 多个不同 LLM 共置，属"多模型共置"而非分阶段调度，建议在其他类别处理 |
| Nova（arXiv 2509.21301） | 排除 | agentic VLM 单 GPU 实时服务，跨阶段 SM 重分配；与本类边界不清，建议边缘 / 实时类别处理 |
| AgentServe（arXiv 2603.10342） | 排除 | 单 GPU agent 负载，green context 槽位；与 Drift 机制重叠，预算外 |
| RAPID-Serve（arXiv 2601.11822） | 排除 | intra-GPU P/D 分离，与 Nexus / semi-PD 机制重叠，预算外 |
| DETShare（arXiv 2603.15042） | 排除 | 训练 + LLM 推理共置，非分阶段调度 |

检索未发现任何把"另一个模型的硬周期截止期任务"与 LLM prefill / decode 一起调度的工作（最后一条检索词的结果均为 LLM 内部分阶段调度）。

## C. 边缘 SoC 上的多模型实时推理与内存带宽争用

#### 检索词

- C1：`Miriam elastic kernels multi-DNN`、`Band coordinated multi-DNN inference heterogeneous mobile processors`、`RT-mDL real-time mixed deep learning tasks edge`、`Heimdall mobile GPU coordination`、`LLM and DNN co-located Jetson Orin concurrent inference scheduling`、`Performance isolation inference edge GPU Green Contexts MPS MIG`。
- C2：`BWLOCK++ integrated CPU-GPU`、`Protecting real-time GPU kernels integrated CPU-GPU SoC`、`memory bandwidth regulation integrated GPU real-time`、`Dynamic Memory Bandwidth Allocation Real-Time GPU-Based SoC`、`memory interference Jetson AGX Orin CPU GPU`、`thesis memory interference CUDA Green Contexts Jetson Thor`、`Fractional GPUs memory bandwidth reservation`。

#### 来源

- 元数据核对：Crossref API（返回会议论文集名与 DOI）、arXiv 摘要页（含 comments 字段）、Dagstuhl DROPS（LIPIcs / OASIcs 论文 PDF 首页）、UNIMORE IRIS 学位论文记录。DBLP 在本次检索中被 bot 防护拦截，未能使用。
- 全文来源：arXiv PDF、Dagstuhl PDF、NSF PAR PDF、作者主页 PDF、UNIMORE IRIS PDF。
- 浏览过的会议范围：SenSys、MobiSys、MobiCom、RTSS、RTAS、ECRTS、ICCAD、IEEE TCAD、WCET workshop；外加 2026 年 arXiv 预印本与学位论文。

#### 会议核对结果

- Miriam：Crossref 记录为 Proceedings of the 21st ACM SenSys（2023），DOI 10.1145/3625687.3625789。
- Band 为 MobiSys'22（Crossref：Proceedings of the 20th MobiSys，DOI 10.1145/3498361.3538948）。
- Heimdall 为 MobiCom'20（Crossref：26th MobiCom，DOI 10.1145/3372224.3419192）。
- BWLOCK++ 完整版为 ECRTS'18（LIPIcs vol.106, 19:1–19:22，DOI 10.4230/LIPIcs.ECRTS.2018.19）；RTAS'17 为其 WiP 短文（DOI 10.1109/RTAS.2017.26）。
- “DART” 在 Miriam 的相关工作中被引用为 [35]，本次未核对到其会议元数据，未纳入。

#### 候选清单（一行一篇）

| 论文 | 会议 / 年份（核对来源） | 处理 | 理由 |
|---|---|---|---|
| Miriam | SenSys 2023（Crossref） | 卡片 | 边缘 GPU 上关键 / 普通双优先级 DNN 共享 GPU，最接近 C1 主题 |
| RT-mDL | SenSys 2021（Crossref） | 卡片 | Jetson 上周期性多 DNN、截止期与错过率指标 |
| Twill | ICCAD 2025（Crossref + arXiv comments） | 卡片 | Jetson Orin 上 DNN 与 LLM 同时推理的唯一候选 |
| Martín et al., Performance Isolation for Inference Processes in Edge GPU Systems | 预印本 arXiv:2601.07600 | 卡片 | 在 Jetson Orin 上直接比较 MPS 与 Green Contexts 的隔离效果 |
| BWLOCK++ | ECRTS 2018（Dagstuhl PDF） | 卡片 | CPU→GPU 内存带宽干扰的代表性机制（C2 基线工作） |
| Aghilinasab et al., Dynamic Memory Bandwidth Allocation | IEEE TCAD 2020（Crossref） | 卡片 | 在 BWLOCK++ 上按 GPU kernel 进度动态放宽 CPU 带宽 |
| Mascareñas González et al., iGPU Memory Interference vs. L2 Cache Locking | WCET 2023 workshop（OASIcs，PDF 首页） | 卡片 | 在 Orin 上测 ARM 核对 iGPU 的 LPDDR5 干扰 |
| Serafini, Demystifying Memory Interference（博士论文第 6 章） | UNIMORE 博士论文，2026（IRIS 记录） | 卡片 | 唯一在 Jetson AGX Thor 上测 CPU–GPU 内存干扰并用 Green Contexts 调节 |
| Band | MobiSys 2022（Crossref） | 排除 | 手机 CPU/GPU/DSP/NPU 间子图放置，无共享 GPU 并发；ACM 全文未取到 |
| Heimdall | MobiCom 2020（Crossref） | 排除 | 手机 GPU 上 AR 多 DNN 协调，硬件与负载离本场景较远，按预算舍弃 |
| Fractional GPUs (Jain et al.) | RTAS 2019（Crossref/CMU 页面） | 排除 | 未取到正文；以独立显存 GPU 的 page coloring 为主 |
| Pipelined Data-Parallel CPU/GPU Scheduling for Multi-DNN Real-Time Inference (Xiang & Kim) | RTSS 2019（Crossref） | 排除 | 早于 2020，且以 CPU/GPU 流水为主 |
| Hetero²Pipe | ICDCS 2025（Crossref） | 排除 | 手机异构处理器流水，非共享 GPU 实时 |
| Edge-Inference Governors Need Memory-Clock State | 预印本 arXiv:2606.16106 | 排除 | 以 DVFS/EMC 频率模型为主，单作者预印本；含 LLM decode 与 GPU co-tenant 的现象，可作补充 |
| MemGuard / MemPol / Memguard-RW / RTSS'18 动态带宽调节分析 | RTAS'13 / RTS 2024 / ISORC'25 / RTSS'18 | 排除 | 只调节 CPU 核之间的带宽，不涉及 GPU |
| Twill 之外的 Orin 多任务论文（OCTOPINF 等） | 预印本 | 排除 | 视频分析吞吐导向，无截止期 |

## D. 机器人双系统的部署

#### 检索词与来源

- WebSearch："DualVLN dual-system vision-language navigation Jetson Thor"；"NavGPT-3 hierarchical navigation runtime arXiv 2610.10787"；"OxyGen unified KV cache VLA multi-task parallelism arXiv 2603.14371"；"Fast-in-Slow dual-system VLA arXiv execution frequency"；"Hume dual-system VLA value-guided System-2 thinking arXiv 2505"；"concurrent VLM planner and VLA policy same edge GPU scheduling interference robot arXiv 2026"（extended）。
- GitHub：NVIDIA/TensorRT-Edge-LLM PR #193，通过 GitHub REST API（`/repos/NVIDIA/TensorRT-Edge-LLM/pulls/193`）取 PR 正文全文，再用网页版看 commit 记录与评审回复。
- 打开过的页面：arXiv abs / html（2610.10787v2、2603.14371、2503.14734v2、2506.01953、2506.07339、2506.01844、2505.21432v4、2512.08186v1、2611.14969）；neurips.cc 虚拟会议页（DualVLN on Thor 170076、OxyGen 149556）；figure.ai/news/helix。
- OpenReview（Ei1Wgdo2I0）PDF 与 API 都返回人机验证页（403），DualVLN on Thor 只拿到摘要。
- 已对照 `research/topics/long-horizon-planner-executor/papers.md` 的 B、C 两节：NavGPT-3、DualVLN on Thor、OxyGen、SmolVLA、RTC、EagleVLA、KerColle 已在其中。本类只在它们的系统面（两模型怎么放、怎么跑、测了什么）上补充，不重复任务层结论。

#### 候选清单（含排除）

| 候选 | 处理 | 理由 |
|---|---|---|
| TensorRT-Edge-LLM PR #193（DualVLN 移植） | 收录 | 唯一一个在 Thor 上直接测同一 GPU 上规划器与执行器共置干扰（同进程/跨进程/流优先级）的来源 |
| DualVLN on Jetson Thor（NeurIPS 2026 workshop） | 收录 | 单块 Thor 跑双系统，指标是观测年龄而非吞吐；仅摘要 |
| DualVLN 原文（Ground Slow, Move Fast，ICLR 2026） | 并入上一张卡 | 只给出 2 Hz / 30 Hz 与远端 4090 部署，单独成卡信息量不足 |
| NavGPT-3 | 收录 | 规划器与 VLA 的重叠 / 顺序执行对比、抢占延迟 p95 |
| OxyGen | 收录 | 同一 π0.5 内语言解码与动作生成共享 KV 并批处理；Thor 实测；MPS 基线 |
| Helix（Figure 博客） | 收录 | 工业界双系统部署：两块板载 GPU 各跑一个系统，7–9 Hz / 200 Hz |
| GR00T N1 | 收录 | System 2 / System 1 的 10 Hz / 120 Hz 与 L40 延迟；部署方式未写 |
| Fast-in-Slow | 收录 | 两系统共享参数、同一 GPU，明确说"硬件不支持两块 GPU 并行" |
| RTC | 收录 | 执行器侧把推理延迟写成 deadline d；远端 4090，注入延迟 |
| SmolVLA 异步推理 | 收录 | client/server 拆分与队列阈值 g 的系统模型 |
| Hume（arXiv 2505.21432） | 排除 | 只给 4 Hz / 90 Hz 与"shared queue"，部署硬件、延迟均未写（§1、§3.3），系统面过薄 |
| EagleVLA（arXiv 2607.12659） | 排除 | 已在 papers.md B2；Thor 上同一 VLA 内的置信度调度，不是两个模型共置 |
| XInfer（Scaling VLA Serving Under Robotic Execution Constraints，arXiv 2611.14969） | 排除（集群级 VLA 服务） | 机器人集群的多请求 VLA 服务（数据中心），不是单机双系统 |
| Robion（Efficient VLA Management and Serving for Robot Factories，arXiv 2609.12075） | 排除（集群级 VLA 服务） | 多 GPU 边缘服务器、多机器人 SLO，不是单 SoC 双模型 |
| KerColle（arXiv 2609.22335） | 排除 | 已在 papers.md C1；关注的是同一 VLA 内 VLM 段与动作段的 kernel 并发 |
| LiteVLA / LiteVLA-H | 排除 | CPU-bound 单模型；LiteVLA-H 已在 papers.md B4 |
