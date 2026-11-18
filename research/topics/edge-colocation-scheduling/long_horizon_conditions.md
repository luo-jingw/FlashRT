# 长时程何时出现、规划器上下文何时变长

- 快照日期：2026-11-17
- 关系：扩展 [long-horizon-planner-executor/survey.md](../long-horizon-planner-executor/survey.md) §2.1（已列 CALVIN 衰减、LIBERO-Long 饱和、RoboCerebra 9.1 子步、BEHAVIOR 6.6 min、π0.5 家务时长），本文不重复这些结论，只补充每个基准的步数/时长/记忆需求，以及规划器每次调用的输入与 token 规模。
- 记号：数字后括号注明原文位置；† = 未核实或只在图中；**判断** = 本综述的推算或归纳，不是原文结论。
- 核对方式：arXiv 编号都下载了 PDF 并核对标题；数值用 `pdftotext` 全文检索取原句。

## 1. 任务级条件：各基准的长度与记忆需求

### 1.1 每个基准的量

| 基准 | 平均 / 上限步数 | 子任务数 | 时长 | 需要记忆或部分可观测 | 出处 |
|---|---|---|---|---|---|
| CALVIN LH-MTLC | 每子任务上限 360 步，30 Hz | 每链 5 个，1000 条链 | 每子任务 ≤ 12 s，每链 ≤ 60 s（**判断**：360/30×5） | 否；子任务完成后才给下一条指令 | 论文 §IV-C2、§III-1（30 Hz）；官方代码 `EP_LEN = 360` |
| LIBERO-Long | 上限 600 步（论文）；openpi 用 520，"longest training demo has 505 steps" | 每任务 2 段 | ≤ 26–30 s（**判断**：20 Hz，`control_freq=20`） | 否 | LIBERO 附录（600）；openpi `examples/libero/main.py`；LIBERO 代码 `bddl_base_domain.py` |
| RoboCerebra | 平均 2,972.4 仿真步（约为已有数据集 6×） | 2 到 20 多个，平均 9.1；平均 3.5 类动作 | 约 149 s（**判断**：基于 LIBERO 平台按 20 Hz 换算） | 是：Memory Exploration、Memory Execution、Random Disturbance、Observation Mismatching 四类（§3.1） | §3.2、图 1(c) |
| VLABench 长时程任务 | 平均 > 500 timesteps，原子任务平均 120 | 文中按 sub-step 计分（M 个），未给均值 † | 未给 † | 部分（隐含意图、常识） | §3（"average episode length exceeding 500 timesteps"） |
| MemoryBench（SAM2Act） | 未给步数 † | reopen_drawer 等 3 个任务，每任务 2–4 个变体，中途按按钮 | 未给 † | 是：按钮后场景 "visually indistinguishable"，破坏 Markov 性 | §3 |
| MemER 真机三任务 | 相机 15 Hz，降采样 2 Hz 输入；整段 "over a thousand frames" | Object Search / Counting / Dust & Replace | "a few minutes of memory"（§1）；1000 帧 @2 Hz ≈ 8 min（**判断**） | 是：遮挡物位置、计数、恢复原位 | §2、§4、§4.1 |
| VoLo（RoboVoLo） | VLM 监控 0.2 Hz，VLA 15 Hz | 4 套件：Common Sense 32 / Memory 30 / Complex References 32 / World Knowledge 32 个任务 | 每 episode 超时 120 s 或 180 s（图 4 标注） | Memory 套件：跨子任务追踪场景状态 | §4、附录 D |
| BEHAVIOR Challenge 2025 | — | 50 个任务（survey §2.1） | 平均 6.6 min，最长任务平均 14 min | 是："holding a radio at the start ... looks identical to holding it at the end" | 第 1 名方案 §1 |
| BEHAVIOR Challenge（官网当期） | — | 100 个任务；有 skill/subtask 标注，未给数量 † | 20,000 条演示共 1,950 h → 平均约 5.9 min（**判断**：除法） | 同上 | behavior.stanford.edu/challenge（2026-11-17 访问） |
| EmbodiedBench | 上限：高层任务 30 步，EB-Navigation 20，EB-Manipulation 15；GPT-4o 平均环境步 EB-ALFRED 16.3、EB-Habitat 13.1、EB-Nav 15.5、EB-Manip 12.9 | Long Horizon 子集：EB-ALFRED 中 "more than 15 steps" | 无物理时长（离散高层动作） | 部分：Long Horizon 子集最难，GPT-4o 从 86% 掉到 64% | §4.1、附录 C、表 7–8 |
| R2R | 约 5 条边、平均 10 m（R2R 原文）/ 9.4 m（RxR 统计）；导航图 4–6 步 | 指令平均 29 词 | — | 部分可观测（未见区域） | R2R §3；RxR §3 |
| RxR | 平均 8 条边、14.9 m；上限 40 m | 指令平均 78 词（英文 en-US 129） | 人类 Follower 平均 132 s | 同上，长度方差大 | RxR §3、表 2 |
| VLN-CE（R2R-CE） | 平均 55.88 个低层动作（R2R 为 4–6）；最好 agent 平均 88 步 | 同 R2R | — | 同上 | VLN-CE §1、§3 |
| NavGPT-3（R2R-CE） | 无周期检查时规划器中位 51 个 VLA 步才拿回控制（90 分位 123） | — | 规划器总推理时间 66.8–425.9 s / episode（表 3） | 部分可观测，需要回溯 | §4.4、表 3 |
| OneTwoVLA 真机 | — | Tomato-Egg 一次试验推理 5 次 | 183 s / 135 s（两条示例轨迹） | 部分 | 附录表 14 下文 |
| LoHo-Manip（VLABench） | — | — | episode 86 s（不加 manager 72 s） | — | 表 8（papers.md） |

### 1.2 条件 → 基准对照

| 条件 | 量化 | 代表基准 |
|---|---|---|
| 子任务数多 | 5（CALVIN）→ 平均 9.1、最多 20+（RoboCerebra） | CALVIN、RoboCerebra、VLABench |
| 单次 episode 长 | 秒级（CALVIN ≤ 60 s、LIBERO ≤ 30 s）→ 分钟级（RoboCerebra ≈ 2.5 min**判断**、BEHAVIOR 6.6 min、最长 14 min） | BEHAVIOR、RoboCerebra |
| 需要记住过去状态 | 遮挡物位置、计数、恢复原位、"开头与结尾外观相同" | MemER、MemoryBench、RoboCerebra Memory、VoLo Memory、GR 1.5 的 Desk Organization / Swap（§5） |
| 部分可观测 | 导航中未见区域；路径 9.4 → 14.9 m、步数 4–6 → 55.88 | R2R → RxR → VLN-CE |
| 出错与重规划 | RoboCerebra 的 Random Disturbance / Observation Mismatching；VoLo 的 recovery / replan；NavGPT-3 回溯 | RoboCerebra、VoLo、NavGPT-3 |

**判断**：CALVIN 与 LIBERO-Long 的"长"只是子任务串联，单段 ≤ 1 min 且不需要记忆；需要规划器长上下文的是后三行（记忆、部分可观测、重规划）叠加分钟级时长的基准。

## 2. 规划器上下文：每次调用放进去什么

### 2.1 各系统的规划器输入

| 系统 | 规划器 | 调用频率 | 每次输入的图像 | 历史形式 | 上下文规模（原文） | 出处 |
|---|---|---|---|---|---|---|
| Hi Robot | π0 同类 VLM | 每 1 s 或用户插话时 | 当前多相机图像 | 无；失败分析写 "lacks memory" | 4090 上 prefill 47 ms + decode 13.2 ms / 步 | §4.1、附录 B.3、C.4 |
| MemER | Qwen2.5-VL-7B 微调 | ~1 Hz（异步） | 每相机最近 N = 8 帧（2 相机，2 Hz 采样）+ 已选关键帧 \|K_t\| ≤ 8 | 视觉关键帧记忆（1D single-linkage 聚类） | 图像 320×180；≈ 66 token/帧 → ≤ 24 张图约 1.6k 视觉 token（**判断**：Qwen2.5-VL 28 px/token）；Long History 32 帧 "inference cost of 1 second" | §3.1、§4、§4.1 |
| VoLo | Claude Opus 4.6（另测 GPT-5.5、Gemini 2.5 Flash、Qwen3-VL-8B） | 0.2 Hz 监控 + 规划点 | BEFORE 图（episode 开始）+ NOW 图 | 文本：子目标列表、上一工具调用及结果；慢上下文存任务记忆 | 每次输入固定大小（**判断**：两张图 + 定长文本）；云 VLM 每次 ~1–5 s | §3（P2）、附录 C.1–C.2、§6 |
| Gemini Robotics 1.5 | GR-ER 1.5 编排器 | 未给 † | 未给 † | 编排器做 success detection 切换子任务 | thinking token 随任务自适应，pointing 最少、video QA 最多（图 16，仅图 †）；ER 2 Preview 官方输入上限 131,072 token | §1、§4、§5；ai.google.dev robotics 文档（2026-11-17） |
| Hi-VLA 编排研究 | Gemini 2.5 Flash（thinking on） | 子任务边界 | 当前观测 | 原始历史 1 / 3 / 5 步 / 全 episode | "the length of history context doesn't affect the performance much"（§4.6、表 7） | §4.6 |
| LoHo-Manip | VLM manager | 每步 | 仅当前帧 | 文本进度摘要（completed primitives + remaining plan） | 理由：历史帧在执行偏离演示时产生分布偏移 | §3 |
| RoboCerebra 基线 | VLM 规划器 | "low-frequency" | "recent observations" | 共享 memory bank 存子目标 | 未给 † | §4 |
| EmbodiedBench | 通用 MLLM（GPT-4o 等） | 每个规划步（平均 2.6–13.2 次 / episode，表 7–8） | 当前帧（500×500）；多帧输入 "struggle to effectively utilize" | 文本：交互历史 + 环境反馈 + 已输出动作 | 输出上限 2048 token | §3、§4.1、§4.3 |
| OneTwoVLA | 单模型（对照用） | 只在关键时刻推理 | 两张图 | 最新推理内容 R | 推理 20 / 100 / 200 token 用时 0.853 / 2.346 / 4.361 s（4090）；一次 183 s 任务推理 5 次共 16 s | 附录表 14 |
| NavGPT-3 | GPT-6 Astra / Claude Opus 5（论文命名） | 每次工具返回一轮；检查频率 16 / 32 VLA 步 | 工具返回：全景 4 视图、地图、VLA rollout 最多 8 个关键帧 | 多轮对话持续累积（provider 侧保留会话） | 每 episode 累计 155.0k–721.0k token（表 3）；每 16 步检查一次升到 2,319.5k（表 5）；案例中每次工具调用后累计 8–9k → 28–30k（图 10–13） | §4.3–4.4 |
| DualVLN（System 2） | Qwen2.5-VL 7B | 2 Hz | 当前 RGB + 历史 RGB | `<history>` 历史帧，数量正文未给 † | KV-cache 复用使 System 2 推理 1.1 → 0.7 s（4090） | §3、§4.4、附录 |
| StreamVLN | LLaVA-Video | 每轮输出 4 个动作 | 滑窗内 8 轮对话 + 压缩记忆 | KV 滑窗 + 3D 体素剪枝记忆 | 记忆 2×196 → 8×196 token 的消融；剪枝减少记忆 token 28%（R2R）/ 22%（RxR）；真机 0.27 s / 4 动作 | §III、§IV |
| Uni-NaVid | Vicuna 类 VLM | 5 Hz | 当前 64 token；短期每帧 4 token（B = 64 帧）；长期每帧 1 token 再合并 | 在线 token 合并 | 长期段按相似度继续合并，次线性增长 | §III-B |
| NaVILA | VILA | — | 当前帧 + 均匀采样历史帧 | 视频历史作 memory | 196 token/帧 × t，t = 8–64 → 1.6k–12.5k（**判断**：乘法） | §III、图 3 |
| BEHAVIOR 2025 第 1 名 | 单模型 System 2 阶段预测 | 每步 | 当前图像 | 阶段编号 + 投票（代替历史） | — | §4.2 |

### 2.2 三种上下文形态（**判断**，由 2.1 归纳）

| 形态 | 每次调用的 token | 随 episode 增长 | 代表 |
|---|---|---|---|
| A. 当前帧 + 定长文本状态 | 定长，约 1–2k（几张图 + 子目标列表） | 不增长 | Hi Robot、VoLo、LoHo-Manip、BEHAVIOR 第 1 名、EmbodiedBench（图像部分） |
| B. 有上界的压缩记忆 | 有上界，约 1.6k–12.5k | 增长到上限后不再增长；上限由关键帧数 / 滑窗 / 合并比例决定 | MemER（≤ 8 关键帧）、StreamVLN（8 轮 + 记忆）、Uni-NaVid、NaVILA（8–64 帧） |
| C. 多轮 agent 对话累积 | 单次 8k → 30k（NavGPT-3 案例） | 线性增长；每次检查重读全部路线上下文 | NavGPT-3；EmbodiedBench 的文本历史部分 |

原文对 B、C 取舍的依据：
- 加长原始历史收益小、代价大：MemER Long History 32 帧推理 1 s，仍比 MemER 平均低 34%（§4.1）；Hi-VLA 编排研究 1 步到全 episode 历史差别不大（§4.6）；EmbodiedBench 多帧输入反而变差（§4.3）。
- 累积对话的代价：NavGPT-3 "Tokens grow much faster than accuracy, because each check re-reads the accumulated route context"（§4.4）。按表 3 平均每轮约 10k（Astra，325.0k / 32.14 轮）到 15k（Opus，721.0k / 47.93 轮）token（**判断**：除法）。

### 2.3 Thinking token

| 系统 | 是否用 thinking | 数量 | 出处 |
|---|---|---|---|
| Gemini Robotics 1.5（GR-ER 1.5） | 是，按任务自适应 | 仅图 †；"fewest tokens for pointing tasks, and most for video QA" | 图 16 |
| Hi-VLA 编排研究 | 是（Gemini 2.5 Flash / Pro） | 未给；thinking 在所有任务上稳定更好，长时程任务受益更大 | §4.2、图 2 |
| OneTwoVLA | 只在关键时刻 | 20 / 100 / 200 输出 token → 0.85 / 2.35 / 4.36 s；> 100 token 时机器人需暂停几秒 | 附录表 14 |
| NavGPT-3 | 是（推理与叙述记入 token） | 计入每 episode 累计 token，未单独拆分 † | 表 3 |
| VoLo | JSON 模式，"reason" 字段一句话 | 每次输出很短（**判断**） | 附录 C.2 |

## 3. 结论：上下文变大或变得不定长的条件

| # | 条件 | 依据 |
|---|---|---|
| 1 | 任务需要回忆已看不到的状态（遮挡物位置、计数、恢复原位），且时间跨度到分钟级 | MemER 三任务（"minutes of memory"，> 1000 帧）、MemoryBench（按钮后场景不可区分）、BEHAVIOR 第 1 名（开头与结尾外观相同）。当前帧方案在这类任务上不足：Hi Robot 失败分析写 "lacks memory"（附录 C.4）；MemER 中 No History 与 Short History "perform poorly as all of the tasks simply require more context"（§4.1） |
| 2 | 规划器以多轮 agent 对话形式运行，工具返回（全景图、地图、rollout 关键帧）留在对话里 | NavGPT-3 每 episode 155k–721k token，单次 8k → 30k（表 3、图 10–13） |
| 3 | 规划器周期性监督执行器，每次检查重读累积历史 | NavGPT-3 检查频率 None / 32 / 16 步：179.2k / 1,529.9k / 2,319.5k token（表 5）；VoLo 为避免这一点把监控上下文限定为 BEFORE + NOW 两图 |
| 4 | 部分可观测导航，路径长且方差大 | VLN-CE 平均 55.88 步；RxR 路径 14.9 m、上限 40 m、长度方差大；历史以视频帧输入（NaVILA 8–64 帧 × 196 token），不压缩时随步数线性增长 |
| 5 | 出错后重规划或回溯，使 episode 长度不定 | NavGPT-3 规划器中位 51 步、90 分位 123 步才拿回控制（§4.4），回溯减少 token 33–41%（表 3）；EmbodiedBench 平均规划步数在模型间 2.6–13.2 次不等（表 7–8）。**判断**：重规划次数决定调用次数和累积长度，长度分布是长尾 |

**判断**：已发表的机械臂规划器大多主动把上下文压到形态 A 或 B（定长或有上界，约 1–12k token）；真正无上界、按 episode 增长到 10⁵–10⁶ token 的只在 agent 式导航规划器（NavGPT-3）中出现。thinking token 是另一项可变量，长度随任务而变，但除 OneTwoVLA 外没有找到给出数值的论文。

## 4. 来源

| 简称 | 链接 | 状态 |
|---|---|---|
| CALVIN | [arXiv 2112.03227](https://arxiv.org/abs/2112.03227) | RA-L 2022（PDF 页眉）；EP_LEN 取自官方代码 |
| LIBERO | [arXiv 2306.03310](https://arxiv.org/abs/2306.03310) | 20 Hz 取自官方代码；520 取自 openpi 代码 |
| RoboCerebra | [arXiv 2506.06677](https://arxiv.org/abs/2506.06677) | 预印本 |
| VLABench | [arXiv 2412.18194](https://arxiv.org/abs/2412.18194) | 预印本 † |
| SAM2Act / MemoryBench | [arXiv 2501.18564](https://arxiv.org/abs/2501.18564) | 预印本 † |
| MemER | [arXiv 2510.20328](https://arxiv.org/abs/2510.20328) | 预印本 |
| VoLo | [arXiv 2606.07723](https://arxiv.org/abs/2606.07723) | 预印本 |
| Gemini Robotics 1.5 | [arXiv 2510.03342](https://arxiv.org/abs/2510.03342) | 预印本；ER 2 输入上限取自官方文档 |
| Hi Robot | [arXiv 2502.19417](https://arxiv.org/abs/2502.19417) | 预印本 |
| Hi-VLA 编排研究 | [arXiv 2606.10267](https://arxiv.org/abs/2606.10267) | 预印本 |
| LoHo-Manip | [arXiv 2604.21924](https://arxiv.org/abs/2604.21924) | 预印本 |
| OneTwoVLA | [arXiv 2505.11917](https://arxiv.org/abs/2505.11917) | ICLR 2026（PDF 页眉） |
| EmbodiedBench | [arXiv 2502.09560](https://arxiv.org/abs/2502.09560) | 预印本 † |
| BEHAVIOR-1K | [arXiv 2403.09227](https://arxiv.org/abs/2403.09227) | 未取到时长数据 |
| BEHAVIOR 2025 第 1 名 | [arXiv 2512.06951](https://arxiv.org/abs/2512.06951) | 预印本 |
| BEHAVIOR Challenge 官网 | [behavior.stanford.edu/challenge](https://behavior.stanford.edu/challenge/index.html) | 当期页面，未确认届次 † |
| R2R | [arXiv 1711.07280](https://arxiv.org/abs/1711.07280) | CVPR 2018（VLN-CE 参考文献） |
| RxR | [arXiv 2010.07954](https://arxiv.org/abs/2010.07954) | EMNLP 2020 † |
| VLN-CE | [arXiv 2004.02857](https://arxiv.org/abs/2004.02857) | ECCV 2020 † |
| NavGPT-3 | [arXiv 2610.10787](https://arxiv.org/abs/2610.10787) | 预印本 |
| DualVLN | [arXiv 2512.08186](https://arxiv.org/abs/2512.08186) | ICLR 2026（iclr.cc 页面）；历史帧数未给 † |
| StreamVLN | [arXiv 2507.05240](https://arxiv.org/abs/2507.05240) | 预印本 |
| Uni-NaVid | [arXiv 2412.06224](https://arxiv.org/abs/2412.06224) | 预印本 † |
| NaVILA | [arXiv 2412.04453](https://arxiv.org/abs/2412.04453) | 预印本 † |
