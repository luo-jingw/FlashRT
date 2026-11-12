# 长时程任务中的规划器 + 执行器：动机、延迟代价与边缘 SoC 上的系统现象

- 快照日期：2026-11-09（第一轮）
- 引用格式：文中用 [papers.md](papers.md) 中的简称，数字后注明原文位置。

## 0. 范围

本综述回答三个问题：
1. **动机**：长时程任务是什么；在长时程任务上，"规划器 + 执行器"是否优于不分层的方案；为什么要分成两个模型（§2）。
2. **代价**：规划器延迟和两模型不同步，怎样变成任务上的损失，用什么指标衡量（§3）。
3. **系统现象**：两个模型放在同一块边缘 SoC（统一内存）上时，有哪些数据中心 LLM 服务没有的现象（§4）；相关系统论文如何从观察走到挑战、机制和评估（§5）。

术语：
- **两模型分层**：独立的 VLM 规划器（或编排器）加 VLA 执行器。
- **单模型分层**：同一个模型先输出子任务或推理，再输出动作。
- **flat**：VLA 直接接收整条任务指令，不做显式的子任务分解。

## 1. 分类框架

三个正交的轴：
- **高层决策放在哪里**：两模型分层 / 单模型分层 / flat。
- **执行器在规划器思考时做什么**：同步（停下等）/ 异步（继续执行旧指令或旧计划）。
- **部署位置**：规划器在云端或服务器 / 规划器与执行器在同一块边缘 SoC 上。

已有工作大多落在"两模型分层 + 同步 + 规划器在云端或独立 GPU"这一格；"两模型 + 同一块 SoC"的组合只有 DualVLN on Thor（仅摘要）和 NavGPT-3 的板载部分涉及。

## 2. 动机：长时程任务与分层

### 2.1 长时程的定义

- **CALVIN ABC→D**：每条链 5 个子任务；flat 方法的成功率随链长迅速衰减，GR-1 从第 1 个的 0.854 降到第 5 个的 0.401，HULC 从 0.418 降到 0.011（GR-1 表 1）。
- **LIBERO-Long（LIBERO-10）**：10 个任务，rollout 上限 520–600 步；已接近饱和，flat π0.5 公布值 92.4，分层方法 95–96（Fast Plans 表 2，LoHo-Manip 表 5）。
- **RoboCerebra**：每任务 2 到 20 多个子步，平均 9.1 个，平均 2,972 个仿真步。
- **BEHAVIOR Challenge 2025**：50 个家务任务，平均 6.6 min，最长任务平均 14 min（BEHAVIOR 2025 第 1 名）。
- **π0.5 家务**：多阶段任务 2–5 min，整间清理 10–15 min（π0.5 引言、§V-A）。

### 2.2 同一执行器下的对照：分层的最强证据

最有说服力的是执行器相同、只加或不加规划器的对照：
- **Hi-VLA 编排研究**：MuJoCo ALOHA 仿真的长时程任务成功率（%，5 个任务各 200 次）flat 25.30 → 简单分层 40.56 → 最佳分层 67.08；真机 ALOHA 15 个水果放对的个数 3 → 9 → 12（表 1）。三种设置 "have the same VLA and input task prompt"（§4.7）。
- **VoLo**：同一个 π0.5，跨套件成功率 12.57 → 41.80（表 2）。
- **Gemini Robotics 1.5**：同一 VLA，单独思考的进度分（部分完成程度的连续分数）最高 44%，加 ER 1.5 编排器常接近 80%（§5）。规划类失败（编排器给出错误计划或错误指令，占全部评测次数）在编排器为 Gemini 2.5 Flash 时 25.5%，为 ER 1.5 时 9%（表 1）：这是两种编排器之间的比较，说明规划器的能力决定分层的收益。

增益集中在长时程或需要推理、记忆的任务上；短时程任务上简单分层与 flat 持平（69.57 对 69.63，Hi-VLA 编排研究表 1）。

### 2.3 增益的条件与证据的薄弱处

- **依赖规划器能力**：VoLo 中规划器换成 Qwen3-VL-8B 只有 19.95，"with weaker VLMs the gain becomes marginal"（§5.4）。Hi-VLA 编排研究发现规划器规模 "matters much less"，开 thinking 稳定有益（表 3）。
- **依赖设计细节**：简单分层只拿到最佳分层约一半的增益；让 VLM 自己决定何时切换子任务最差（Hi-VLA 编排研究表 5）。Fast Plans 的两个变体与 π0.5 持平或更低，增益来自计划与执行之间的对齐机制（表 2）。
- **绝对成功率低**：RoboCerebra 最好 16.55%，真值计划也只有 25.16%（表 3、4）。
- **基线弱或协议不匹配**：Agentic Robot 的 61.6 远低于 flat π0.5 公布值 92.4；Fast Plans 承认 π0.5 一行协议不完全匹配。
- **真机样本小**：每格 10–20 次，且常常只有图（Hi Robot、π0.5、OneTwoVLA）。

### 2.4 反证：单模型同样有效

- **π0.5**：同一模型内做子任务预测，优于人类高层 oracle；GPT-4 零样本高层最差（§V-E，图 13）。
- **OneTwoVLA**：单模型推理加行动，平均 87%，比 Gemini 2.5 Pro + π0 的双系统高 24%（§4.1）。
- **VLA-OS**：Integrated（单模型内规划）与 Hierarchical "largely comparable"，都优于只出动作的 VLA。
- **BEHAVIOR 2025**：前两名都是单模型（第 1 名与 Openpi Comet），但都没有做分层消融。

注意点：输给单模型的双系统，规划器多是未用机器人数据适配的通用大模型（OneTwoVLA 的 Gemini 2.5 Pro、π0.5 的 GPT-4）。文献支持的结论是两条：显式的高层子任务或进度表示优于纯 flat；规划器是否需要成为独立的大模型，取决于任务是否需要开放词汇理解、世界知识或长记忆（VoLo 的 Common Sense / Complex References 套件，Gemini Robotics 1.5 的规划失败下降）。

### 2.5 为什么分成两个模型

论文给出的理由：
- 复用大规模预训练 VLM 的推理能力，避免在窄的控制数据上微调造成遗忘（HiVLA）；
- 两层运行频率不同（Hi Robot §4；MemER 高层约 1 Hz）；
- 规划器与执行器可以单独替换（LoHo-Manip 表 7 换执行器）；
- VLA "primarily optimized for action output"，规划能力有限（Gemini Robotics 1.5 §5）。

反方理由：两层 "are not aware of one another's capabilities"（Hi Robot §6）；规划器可能输出执行器做不了的内容（OneTwoVLA §1）。

## 3. 规划器延迟与不同步的代价

### 3.1 延迟变成任务代价的三条途径

1. **同步：机器人停下等**，代价是完成时间变长。
   - Fast Plans：手臂执行动作的墙钟时间占比 28.7% → 65.6%（表 3A）。
   - SmolVLA：异步比同步完成时间短约 30%（13.75 → 9.70 s，§4.6）。
   - LoHo-Manip：episode 86 s，不加规划器 72 s（表 8）。
2. **异步：执行器按旧指令或旧观测继续动作**，代价是成功率下降，幅度取决于执行器是否按部署时的刷新周期训练。
   - Leap-VLA：同频训练的策略在 2 s 刷新时 61.4，按部署时钟训练 95.5（表 III）。
   - Slow Brain, Fast Planner：直接沿用旧建议的方式在 2 s 延迟后崩溃，融合方式到 5 s 仍高于 80%（图 6）。
   - VLALagBench：移动目标在 100 ms 同步延迟下 set-down 成功率 71.5% → 40.5%，近静止目标在 200 ms 下几乎无损（§4.2）。
3. **发现问题到纠正之间的滞后**：NavGPT-3 真机上规划器与 VLA 重叠执行时修复时间 7.9 s、顺序执行 12.8 s，按过时指令执行的命令 1/30 对 8/30（表 10）。

### 3.2 指标

与任务结果直接挂钩、且最直白的：
- **墙钟完成时间**：环境不暂停、失败记为上限（SmolVLA、AsyncVLA、NavGPT-3）。
- **执行占比（motion duty cycle）或空等比例**：执行动作的墙钟时间 / 总时间（Fast Plans）；VLALagBench 的 idle steps。
- **固定时间内完成的任务数**（SmolVLA）；**task throughput** = 完成比例 / 时长（RTC）。
- **注入延迟下的成功率曲线**（Leap-VLA、Slow Brain、VLALagBench、CloudEdgeVLA）；CloudEdgeVLA 把它汇总成 delay-retention AUC。
- **事件到修复的时间**（NavGPT-3）。

间接的：
- **observation staleness / age**（VLALagBench、DualVLN on Thor），DualVLN on Thor 发现它与导航质量不单调对应；
- **reaction time**（EagleVLA：环境变化到机器人响应）；
- **action / control frequency**（OxyGen、EagleVLA），只是截止期满足度的代理。

### 3.3 文献中的空白

- 没有工作测过**规划器延迟 5–7 s、超过检查间隔时**，周期性子目标检查在操作任务上的成功率或完成时间：Leap-VLA 只到 2 s；Slow Brain 是导航、用云端 VLM；NavGPT-3 的检查周期以步数计。
- **子目标实际完成到执行器收到下一个子目标之间的滞后**，没有论文定义或测量过。
- **同一块 SoC 上规划器延迟与执行器截止期违约的联合代价**没有被测量：OxyGen 只报动作频率；DualVLN on Thor 只有摘要，且仿真在模型计算期间冻结场景。
- 主流基准不计推理延迟：BEHAVIOR 用与硬件无关的仿真时间，vla-eval 阻塞式评测、只统计成功率。
- 秒级 VLM 规划器的事件触发与周期调用的闭环对比没有人做过；只有动作层的 VLA-Corrector 有评测。

## 4. 边缘 SoC 上的系统现象

### 4.1 SM 划分保护算力，保护不了带宽

- **Ripple**：DGX Spark 上 LLM decode 使共用统一内存的 CPU 任务慢 1.7–2.5×，H200 上 "no impact"；把 LLM 从 48 个 SM 缩到 8 个 "has very limited effect"（§2.2，图 3a）；合计 DRAM 吞吐 190–257 GB/s，标称 273 GB/s。
- **EdgePace**：Orin Nano 上算力型背景任务的影响可以忽略，访存型背景使相机任务 p95 1.52×、超时 72.78%（§1，图 1）。
- **XSched**：Orin 上 GPU 与 NPU 算力独立，但 "may compete for the memory bandwidth and power supply"（§7.2）。
- **HaX-CoNN**：Orin / Xavier 上 GPU 与 DLA 并发争用 DRAM，不考虑争用的映射最多降 70%。
- **iGPU L2 锁定**：AGX Orin 上 CPU 压 LPDDR5 的干扰约占 iGPU 卷积 WCET 的 26%（§4.3）。
- 没有找到在 Jetson LPDDR 上做通道或 bank 划分的公开工作；SGDRC 只覆盖 GDDR 独显。

### 4.2 内存时钟（EMC）是看不到的状态

- **Edge-Inference Governors**：Orin 上不考虑 EMC 档位的延迟模型在紧截止期下超时 25–28%，考虑后 ≤ 0.9%；超时成簇；作者把 actmon / BPMP 的 EMC 钉频策略列为未解释的原因（表 XI）。
- **Arya & Simmhan**：AGX Orin 上 EMC 3200 → 665 MHz，LLM 延迟约 +370%（§3.4）。
- **DVFSLM**：EMC 0.204 GHz 时 "severe QoS degradation"（§5.3）。
- **Hydra**：刻画了 Thor 上的 LLM 分阶段特性，但没有 Thor 的带宽、功耗模式和 warm-up 数据（§VI）。

### 4.3 空闲之后变慢：没有找到直接测量

没有论文测量过"SoC 空闲几分钟后，解码在约一分钟内变慢"的现象。最接近的两项：
- Edge-Inference Governors：高 EMC 档下 L2 常驻 kernel 的标准差扩大约 10 倍，GPU 时钟与温度不变，原因未查明（§VIII-A）；
- StarSD：独显上空闲间隙后的每 token 时间变慢在几十毫秒量级，锁频后仍存在（§5.3），与分钟级的现象不是同一类。

DRAM self-refresh 退出、actmon 升档滞后、SMMU / TLB 冷启动都没有公开测量。

### 4.4 数据中心 LLM 服务为什么没有这些问题

- **带宽争用**：CPU 用独立 DDR，GPU 用 HBM，每 GB 带宽是 DGX Spark 的 13–20 倍（Ripple 表 1）；H200 上 CPU 任务不受 decode 影响（Ripple）；数据中心的 CPU 侧共置问题 "not slower GPU kernels"（Host-CPU 共置干扰）。
- **硬件隔离**：数据中心 GPU 有 MIG 划分 L2 与内存控制器，有 DCGM / Nsight 的带宽计数器；DGX Spark 上 MPAM 未开启、DCGM 的 DRAM 指标为空（Ripple §2.2）。
- **内存时钟**：HBM 时钟 "not a deployment-time knob"（Edge-Inference Governors §XII）；服务器负载稳定、时钟高位。
- **功耗**：每块 GPU 有独立的板级功耗上限；边缘 SoC 上 CPU、GPU 与加速器共享预算，Orin Nano 两模型并行时 GPU 频率约减半（边缘隔离实测）。
- **指标**：服务的 SLO 是大量请求上的 TTFT / TPOT 分位数；周期截止期对单次超时敏感，且超时成簇（Edge-Inference Governors §VII）。

## 5. 系统论文的写法：观察 → 挑战 → 机制 → 评估

精读 Ripple、EdgePace、XSched、Orion、Edge-Inference Governors，共同模式如下：
1. **先给一个反直觉的对照测量**，排除显而易见的解释：Ripple 的 48 → 8 SM 无效加 H200 对照；EdgePace 的算力型与访存型背景对比；Orion 的同类与互补 kernel 共跑对比（BN2d + BN2d 1.08×，Conv2d + BN2d 1.41×，表 2）；Governors 的 EMC-blind 与 EMC-aware 拟合对比。
2. **挑战写成"缺少某个可观测量或控制手段"**：没有按请求者区分的带宽计数器（Ripple）、看不到 EMC（Governors）、硬件调度器不能抢占（XSched），并用一个数字说明朴素替代为什么失败（EdgePace 的带宽阈值告警 93.84% 是误报）。
3. **机制由三件事组成**：把干扰方切成可暂停的单元（quantum、层、命令队列）；一个代理信号（模型推导的字节量、前台 PMU、离线资源标签、短探测）；一条准入规则，在时间上限制干扰方。
4. **主指标来自"谁受害"**：受害方有周期截止期时，主指标是超时率并设上限（Ripple 2%，Governors τ = 2%），同时报告干扰方保留的吞吐；受害方是请求流时用 p99 或相对独立运行的倍数（Orion、XSched）。
5. **截止期按受害方的分布设定并扫描**（Ripple 取共跑 p99 的 k 倍，k = 1.1–2.0）；基线里一定有朴素的隔离手段（MPS、暂停、只调度一种加速器）。

刻画类论文（VLA across XPUs）的写法：每条 "Finding #N" 一句结论配一张单指标图，刻画在前、优化在后，优化的动机直接引用前面的 Finding（见 [notes/2026-vla-xpu-characterization.md](notes/2026-vla-xpu-characterization.md)）。

## 6. 文献自述的开放问题

- 高层缺少记忆，难以完成跨房间任务（Hi Robot 附录 C.4；π0.5 局限）。
- 两层互不了解对方能力（Hi Robot §6；OneTwoVLA §1）；两系统的交互仍然有限，需要双向通信与失败恢复信号（RoboCerebra 结论）。
- 剩下的失败主要在执行器的灵巧性（MemER；BEHAVIOR 2025 第 1 名；Gemini Robotics 1.5 §7）。
- 只在静态环境测试、假设延迟不关键（Hi-VLA 编排研究局限）；VLM 单次延迟 "bounds reaction time and may miss fast failures"（VoLo §6）。
- 何时推理、何时切换子任务仍靠启发式（OneTwoVLA §5；Hi-VLA 编排研究表 5）。
- 事件触发调用没有评测（NavGPT-3 §4.4；Slow Brain future work；LiteVLA-H §3.2）。
- 仿真在推理期间冻结场景，复现不了真实的场景变化（Slow Brain 局限；DualVLN on Thor 摘要）；真实延迟会抖动，而评测把它当作固定整数（Understanding Asynchronous Inference §7）。
- 没有按请求者区分的带宽计数器（Ripple §2.2）；EMC 钉频策略、内存控制器仲裁等原因未解释（Edge-Inference Governors 表 XI）；只测过 CPU 流式写这一种干扰，GPU 共跑者未测（同上 §XII）。
- Thor 的带宽、功耗模式与 warm-up 没有刻画（Hydra §VI）。
- 把 memory-bound 的 decode 与计算密集负载共置列为后续工作（Orion §7）；Robion 假设 VLM 为 compute-bound。

## 7. 入门阅读顺序

1. Hi-VLA 编排研究：同一 VLA 下 flat、简单分层、最佳分层的系统比较，以及分层在什么条件下有用。
2. π0.5 与 OneTwoVLA：单模型的反证，必须回答"为什么不做成一个模型"。
3. VoLo 与 Gemini Robotics 1.5：同一执行器加规划器的增益，以及增益随规划器能力的变化。
4. VLALagBench 与 Leap-VLA：延迟换算成控制步的方法，以及秒级刷新下执行器训练方式的决定作用。
5. NavGPT-3：唯一在真机上测量重叠与顺序执行的修复时间与过时指令的工作，板载 VLA 跑在 Thor 上。
6. Ripple 与 EdgePace：统一内存上 SM 缩减无效的对照测量，以及用准入在时间上限制干扰方。
7. Edge-Inference Governors：EMC 作为看不到的状态。
8. VLA across XPUs：刻画类论文的 Finding 写法。
