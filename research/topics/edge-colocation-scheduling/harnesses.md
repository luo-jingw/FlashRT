# Planner–Executor Harness 调研（长时程机器人任务）

- 快照日期：2026-11-17
- 范围：VLM/LLM planner + 低层 policy/VLA executor 的系统；关注 harness（调用方式、规划器输入、时延与质量）。
- 元数据：标题、编号、发表状态在 arXiv 页面或项目页核对；与 [long-horizon-planner-executor/papers.md](../long-horizon-planner-executor/papers.md) A/B 节一致的条目沿用其已核实数字。
- 数字均取自论文正文，括号内为位置（§ 章节、表、图、附录）。标 † 为未核实项；"判断" 为本文推断，不是论文结论。
- "代码" 列：项目页上有 Code 链接即记为"有"，仓库内容未逐一核对（†）。

## 1. 总表

| 系统 | 发表状态与链接 | 规划器（模型、大小、运行位置） | 执行器 | 调用方式 | 规划器输入与上下文 | 思考/推理 token | 报告的时延 | 质量指标 | 代码 |
|---|---|---|---|---|---|---|---|---|---|
| **VoLo / VoLoAgent** | 预印本 2026，[arXiv 2606.07723](https://arxiv.org/abs/2606.07723) | Claude Opus 4.6（消融：GPT-5.5、Gemini 2.5 Flash、Qwen3-VL-8B）；云 API（§5.1、表 2） | π0.5 为 "interruptible tool"，另有 SAM3/Molmo2 感知与 GraspGen 抓放原语；15 Hz（§5.1） | **异步**：机器人运动与 agent 推理并行（P1，§4）；监控**周期 0.2 Hz（5 s）**；recovery 时**事件触发**进入 deliberation，执行器被 idle（P3） | 双上下文（P2，§4）：monitor context = 当前前视图 + 当前 subgoal + 近期决策；deliberation context = task memory、scene history、tool catalog，仅在规划点读取。外部 memory 存子目标、最终目标、初始场景。上下文长度未给出 | 未说明是否开启 extended thinking † | 单次调用 "∼1–5 s for cloud VLMs"，限制反应时间（Limitations） | RoboVoLo 跨套件 SR：π0.5 12.57 → Full 41.80；Only-VLA 34.97；规划器换 Qwen3-VL-8B 19.95（表 2）；VLM 失败计数 Opus 4.6 最少（图 6，n=90） | 有：[NVlabs/VoLoAgent](https://github.com/NVlabs/VoLoAgent) † |
| **NavGPT-3**（导航） | 预印本 2026，[arXiv 2610.10787](https://arxiv.org/abs/2610.10787) | GPT-6 Astra（Codex scaffold）或 Claude Opus 5（Claude Code scaffold）；云（§4 setup） | NavGPT VLA（Qwen3-VL 4B/8B + MLP 头，8 个 waypoint）；真机 4B 在 Jetson Thor 上 FP8+TensorRT（§4.8.1） | 工具调用式：Planner 调 `navigate_by_instruction` 委托 VLA 跑一段 rollout，返回后再决策（表 4）；OS 式 runtime 支持 **overlap**（后台线程 review 时 VLA 继续跑）与优先级中断（§3.1.3） | harness 构造的上下文：最多 8 张 route keyframe、终点全景、observed top-down map、node table、执行状态（表 4）；scaffold 自动压缩长上下文；每 episode 155–721 k tokens（表 3） | 未单列；Planner 为推理模型，运行在各自 provider scaffold 中 † | Planner 每 3–19 s 决策一次，VLA 0.5–1 s/步（§1）；表 3：reaction 3.3–19.3 s（无 VLA）对 0.78 s（有 VLA）；真机 p95 halt 1840 → 180 ms（表 10） | R2R-CE 子集 SR 72→82（Astra，表 3）；真机 Go2 revision SR：sequential+queued 73.3 → overlapping+priority 83.3；repair 12.8 → 7.9 s；stale command 8/30 → 1/30（表 10，各 30 次） | 项目页链接 [metacognitionai/NavGPT-3](https://github.com/metacognitionai/NavGPT-3)；论文写 "will release" † |
| **Gemini Robotics 1.5 Agent** | 预印本 2025，[arXiv 2510.03342](https://arxiv.org/abs/2510.03342) | GR-ER 1.5（Gemini 系，规模未公开）作 orchestrator；云 API †（§2.1） | GR 1.5 Thinking VLA，作为 orchestrator 的 "specialized tool"，接收自然语言指令（§2.1）；运行位置未说明 † | 工具调用式：orchestrator 分解任务、做 success detection 决定何时切下一步（§2.1）；调用周期未披露 | 用户输入 + 环境反馈；可调 web search 等数字工具；"Desk Organization"/"Swap" 需记住场景状态（§5）。上下文长度未披露 | **是**：ER 1.5 有 thinking（可设 thinking budget，§4.3）；VLA 也在动作前生成 thinking trace（§2.1、§3.3） | 端到端时延未报告；success detection 实时评测按 5 Hz 采样并模拟推理时延，"stale success predictions quickly become irrelevant"（§4.2，图 13 仅图） | 规划类失败 25.5%（Flash 编排）对 9%（ER 1.5）；总失败 44.5% 对 22%（表 1）；复杂任务 progress score 近 2 倍（§5，图仅） | 无（ER 1.5 以 API 提供） |
| **Hi Robot** | 预印本 2025，[arXiv 2502.19417](https://arxiv.org/abs/2502.19417) | PaliGemma-3B 微调的 high-level VLM；本地 1–2 块 RTX 4090（附录 B.2） | π0（PaliGemma-3B + flow）低层，action chunk（附录 B.3） | **周期 + 事件**：每 1 s 重算子任务，或用户插话时立即重算（§4.1–4.2）；作者提及异步为未来方向（§6） | 当前多视角图像 + 用户 prompt ℓt（含语音转写）；无历史图（§4.1） | 否（直接输出子任务语言 + 可选语音回复） | 高层单步：4090 47 ms prefill + 13.2 ms decode；H100 17.3 + 5.7 ms；低层 73 ms 板载 / 86 ms 板外+WiFi（附录 B.3） | Instruction Accuracy 比 GPT-4o 高层平均高 40% 以上（§5.3，图 5 仅图） | 无官方实现 † |
| **MemER** | 预印本 2025，[arXiv 2510.20328](https://arxiv.org/abs/2510.20328) | Qwen2.5-VL-7B 微调；独立 server（§3.3） | π0.5（DROID 微调），每 chunk 15 步、执行 8 步后重规划（§4.1） | **异步**：πh ≈1 Hz、πl ≈2 Hz，各自 server；πh 计算时 πl 用最新子任务（§3.3，沿用 Hi Robot 做法） | 最近 N=8 帧（2 Hz 子采样，base+wrist）+ 经 keyframe filter 选出的历史 keyframe（记忆检索）（§3.2、§4.1） | 否 | 32 帧长历史推理约 1 s，"approaches the limit"（§4.1）；GPT-5 / GR-ER 1.5 API 10–15 s（§4.1 Q2） | 三项记忆任务 >90%（§4.1，表 1）；Long History 平均低 34%；**API 规划器 10–15 s 时延在闭环中 "complete failures"**（§4.1 Q2） | 有：[memer-policy/memer](https://github.com/memer-policy/memer) † |
| **Agentic Robot** | 预印本 2025，[arXiv 2505.23450](https://arxiv.org/abs/2505.23450) | GPT-4o 做一次性子目标分解（§2.2）；verifier 为 Qwen2.5-VL-3B LoRA（§2.4）；planner 云 API | OpenVLA，10 Hz（§2.5） | planner：任务开始时调用一次；verifier：**周期 0.5 Hz（每 20 帧）**，失败时触发 recovery（抬升夹爪后重评估）（§2.5、§3.1） | planner：指令 + 可选初始图 I0 + skill library + few-shot；verifier：滑动窗口近期 (third-person, wrist) 图像对（式 10） | 否 † | 每 10 步 vs 每 50 步验证，LIBERO-Long 每 episode 用时差 >15 s（§3.4） | LIBERO-Long 61.6（表 1）；验证间隔 50 步时 61.8 → 55.8（§3.4，图 4） | 有：[Agentic-Robot/agentic-robot](https://github.com/Agentic-Robot/agentic-robot) † |
| **RoboCerebra（Planner+VLA 基线）** | 预印本 2025，[arXiv 2506.06677](https://arxiv.org/abs/2506.06677) | GPT-4o / Qwen2.5-VL / LLaVA-Next-Video-7B，及微调 VLM（§5.1）；位置未说明 | OpenVLA 微调（亦测 π0-FAST，表 6） | 规划器先生成步骤级 subgoal 存入 memory bank；之后**周期性**看近期观测监控进度，完成/偏离时更新子目标（§4.2）；周期值正文未给出 † | 低频视频观测 + plan memory（§4.2） | 否 | 未报告 | GPT-4o+OpenVLA 16.04 对 OpenVLA* 4.57；GT-plan 25.16（表 3、表 6） | 有：[qiuboxiang/RoboCerebra](https://github.com/qiuboxiang/RoboCerebra) †（是否含该基线未核实） |
| **LoHo-Manip** | 预印本 2026，[arXiv 2604.21924](https://arxiv.org/abs/2604.21924) | LoHo-Manip-4B（Qwen3-VL 初始化，冻结视觉编码器）；真机 A6000（附录 C.3） | π0.5 等 VLA，以 2D visual trace + 子任务为条件（§2.3） | **receding-horizon 周期**：仿真每 100 个执行器步调用一次；真机 planner ≈2 Hz、executor ≈10 Hz（附录 C.3） | **只看当前帧** + 文本记忆（completed/remaining 列表），不输入历史帧（§2.3） | 否 | 仿真单 episode 86 s 对不加 manager 72 s（表 8） | VLABench π0.5 0.24 → 0.39（表 4） | 项目页无代码链接 |
| **Hi-VLA 编排研究** | 预印本 2026，[arXiv 2606.10267](https://arxiv.org/abs/2606.10267) | Gemini 2.5 Flash-Lite/Flash/Pro（thinking on/off）；跨栈 Qwen3-VL 4B/8B/32B（§4.2、§4.8） | GR On-Device 系列；跨栈 π0.5（§4.3、§4.8） | 统一循环：VLM 出命令 → VLA 运行直到 termination β（success detector 或固定时长）（Alg. 1）；推荐固定时长 4–8 s（§4.4，图 4） | memory 消融：历史 1/3/5 步或整段、摘要/反思、跨 episode 记忆（§4.6，表 7） | **是**：thinking on 全面优于 off，长时程收益更大（§4.2，图 2） | 未报告；作者明确假设 "latency is not a critical performance factor"（Limitations） | 长时程 flat 25.30 / 简单分层 40.56 / 最佳分层 67.08（表 1） | 未见代码链接 † |
| **Tri-System π0.5（Critic in the Loop）** | 预印本 2026，[arXiv 2603.05185](https://arxiv.org/abs/2603.05185) | π0.5 的 subtask generation 模式作 System 2（同一模型，§3.2）；Critic 为 Florence-2-base 0.2B（§4） | π0.5 action 模式（System 1），10 Hz chunk（图 3） | **事件触发**：System 2 休眠，仅在 Critic 判定完成/失败/停滞时唤醒；Critic 20 Hz 异步监控；System 1 在 action buffer 用尽时同步（§3.4） | 多视角当前观测 + Critic 提供的 short-term memory（事件编码）（§3.2） | "on-demand thinking"，即按需调用 System 2（§4） | 无数值；双系统基线 "subtasks at every action chunk led to substantial latency"（§4） | Arrange Tableware 0/10、6/10、9/10（表 1）† | 未见 |
| **Fast Plans（Faithful Actions）** | 预印本 2026，[arXiv 2609.30833](https://arxiv.org/abs/2609.30833) | PaliGemma（Gemma-2B）waypoint planner，与 action expert 同一 π0.5 式模型（图 1） | 同模型 action expert，每段约 0.5 s，30 Hz（§6.2） | **同步阻塞**：仿真在整段 plan 执行完才请求新 plan；真机在第一段执行完即请求（receding horizon）（§4.1） | 当前图像 + 指令 + 状态（图 1） | 否（waypoint token） | plan 时延 1094 → 125 ms，往返 1266 → 293 ms，action expert ≈160 ms（§6.2，表 3/表 5） | motion duty cycle 28.7% → 65.6%（表 3）；真机 SR 48/60、46/60、51/60 基本持平（表 5）；LIBERO-Long 96.2（表 2） | 未见 |
| **DualVLN**（导航） | ICLR 2026（项目页/ICLR 页列出 †），[arXiv 2512.08186](https://arxiv.org/abs/2512.08186) | Qwen2.5-VL-7B 微调，输出 pixel goal + latent goal；远端 RTX 4090（§3.1、§4 Real-world） | 轻量 DiT 轨迹策略（30 Hz）+ MPC 跟踪（§1、§4） | **异步**：System 2 2 Hz，System 1 30 Hz，"a fresh trajectory is always available"（§4） | 当前 RGB + 历史 RGB（prompt 中 `<history>`，附录）；可自主调整视角 | 否 | System 2 KV-cache 复用后 1.1 → 0.7 s；System 1 生成 32 条轨迹 0.03 s（§4 Real-world） | R2R-CE Val-Unseen SR 64.3、RxR-CE SR 51.8（表 1）；Thor 版：p95 观测年龄 2.91 → 0.99 s 时 SR 0.556 → 0.458（[OpenReview](https://openreview.net/forum?id=Ei1Wgdo2I0) 摘要）† | 有：InternNav（论文首页标注）† |
| **Helix (Figure)** | 公司技术博客 2025，[figure.ai/news/helix](https://www.figure.ai/news/helix)，无论文 | S2：7B 开源 VLM，7–9 Hz；机载低功耗嵌入式 GPU（与 S1 各占一块） | S1：80M cross-attention encoder-decoder，200 Hz | **异步**：S2 作为后台进程持续更新共享内存中的 latent vector，S1 每步读最新值 | 未披露 | 否（输出 latent，不是文本） | 仅给频率 | 未给出定量 SR † | 无 |
| **Inner Monologue**（含前身 SayCan） | CoRL 2022 †，[arXiv 2207.05608](https://arxiv.org/abs/2207.05608)；SayCan [arXiv 2204.01691](https://arxiv.org/abs/2204.01691) | InstructGPT（仿真/桌面）或 PaLM 540B（厨房）；SayCan 用 PaLM 540B + value function 打分；云/内部服务（§4） | 预训练语言条件技能（BC-Z/RT-1 类）、CLIPort 等 | **事件触发（技能边界）**：每执行完一个技能后把 success detection / scene description / 人类反馈作为文本追加到 prompt 再生成下一步（§3） | 全部文本历史（已执行技能 + 反馈）；无图像直接输入（§3.3） | 否（2022 年 LLM） | 未报告 | 真机厨房 IM 加成功检测 60.4% 对 SayCan 30.8%（表 3）；SayCan 规划 84% / 执行 74%（表 2） | SayCan 有桌面开源版（say-can.github.io）；IM 无 |

## 2. 共性 harness 模式

1. **周期监控 + 事件重规划是主流调度**。纯周期：Hi Robot 1 s、MemER ≈1 Hz、LoHo-Manip ≈2 Hz/每 100 步、Agentic Robot verifier 0.5 Hz、VoLo 0.2 Hz；事件优先：Tri-System（Critic 触发）、Inner Monologue（技能结束触发）、Hi Robot（用户插话）。判断：周期值都是凭经验选定，跨度为 0.2–2 Hz；只有 Agentic Robot（§3.4）和 Hi-VLA 研究（§4.4）做了周期扫描。
2. **规划器把执行器当作可中断的工具来调用**（VoLo、NavGPT-3、GR 1.5）。规划器决定何时委托 VLA、何时收回控制，返回值是结构化证据（keyframe、执行状态、地图）。NavGPT-3 再往上加了一层 OS 式 runtime，管理 motion authority 和优先级中断。
3. **规划器与执行器的耦合方式分三种**：同步阻塞（Fast Plans、LoHo-Manip 仿真、Agentic Robot 仿真；判断：后两者依据是 episode 用时随调用频率增加）、异步共享最新指令或 latent（Hi Robot、MemER、DualVLN、Helix、VoLo）、事件唤醒（Tri-System）。
4. **上下文要控制在亚秒级推理以内**。常见做法有：只看当前帧加文本进度（LoHo-Manip）、近期帧加检索出的 keyframe（MemER）、双上下文（VoLo 的 monitor/deliberation 两层）、harness 压缩成地图和节点表（NavGPT-3）。云端前沿模型（Opus、GPT-5.x、Gemini）多用作低频编排器，每次调用 1–19 s；端侧 3–8B 模型承担 1–2 Hz 的闭环规划。
5. **监控者与规划者常拆成不同模型**：Agentic Robot 用 3B verifier，Tri-System 用 0.2B Critic，GR 1.5 由 orchestrator 兼做 success detection。判断：这样做是为了让高频检测不受大模型时延限制，VoLo 的 Limitations 也写明需要 "fast local monitors"。

## 3. 报告了"规划器时延 → 任务质量"关系的系统

| 系统 | 证据 | 位置 |
|---|---|---|
| NavGPT-3 | 唯一做了受控 2×2 对比（sequential/overlap × queued/priority）：SR 73.3 → 83.3，repair 12.8 → 7.9 s，stale 8/30 → 1/30 | 表 10 |
| MemER | API 规划器 10–15 s 时延，闭环 "complete failures"；32 帧历史约 1 s，已接近可容忍上限 | §4.1 |
| Agentic Robot | 验证间隔 20 → 50 步，LIBERO-Long SR 61.8 → 55.8；10 步时 episode 用时多 >15 s | §3.4，图 4 |
| Hi-VLA 研究 | 执行时长（即 VLM 调用间隔）扫描，过长会超时，推荐 4–8 s；同时声明未建模 latency | §4.4 图 4；Limitations |
| Fast Plans | plan 时延 1094 → 125 ms，duty cycle 29% → 66%，真机 SR 基本不变（时延影响用时，不影响成功率） | 表 3、表 5 |
| GR 1.5 | success detection 实时评测模拟推理时延，结论是陈旧预测迅速失效（只有图，没有数值） | §4.2，图 13 |
| DualVLN on Thor | 观测年龄与 SR 的权衡（仅摘要）† | OpenReview |
| VoLo | 只有定性说明：1–5 s 云端时延限制反应时间，可能漏掉快速失败；没有扫描 | Limitations |

判断：仅 NavGPT-3 和 Agentic Robot 在同一系统内改变时序变量，并同时报告 SR 与用时；其余或只有定性结论，或只扫描调用间隔、没有扫描时延本身。所有系统都没有在固定调用周期下单独注入规划器时延并测量 SR，只有 papers.md B1 中的 Leap-VLA、Slow Brain Fast Planner 做过这类注入实验，但它们是单模型或导航设置。

## 4. 补充说明

- **未单列**：Code as Policies（[arXiv 2209.07753](https://arxiv.org/abs/2209.07753)，Codex code-davinci-002 一次性生成调用感知与控制 API 的程序，执行器不是 learned policy）；π0.5 / OneTwoVLA（同一模型内先预测子任务再输出动作，不存在独立 harness）；GR00T N1（System 2 的 VLM 特征直接供 System 1 使用，没有语言规划步骤）。
- **VoLo 状态机**：每个监控步骤从 {continue, next_subgoal, recovery} 中选一项；recovery 时进一步从 {continue, replan, rewrite VLA 指令, grasp, place} 中选择（§4）。
- **NavGPT-3 的 overlap 与 priority**：overlap 指 VLA 执行期间后台线程用 LM 审查路线；priority 指 LiDAR 事件直接撤销命令过滤器的 motion authority，无需等待 Planner（§3.1.3、§4.8）。
- **Gemini Robotics 1.5**：两层模型的运行位置与调用周期均未公开，表中相应字段标 †。
