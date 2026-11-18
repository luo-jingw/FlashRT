# 推理时延 → 任务质量：如何量化，以及"都满足时限"时用什么单一指标排序

调研问题：机器人/具身 AI 与实时系统论文如何把 planner/policy 的推理时延映射到 task quality；当所有系统都满足时限时，用什么单一指标排序。
标注：† = 未能从原文核实（来自摘要页/二手来源/记忆）；"判断" = 本综述的推断。数字均来自原文页面（arXiv HTML/摘要页），位置注在括号里。

## 结论先行

- 机器人论文几乎都不直接画"success vs latency"单条曲线就结束，而是**把时延折算进一个带时间的质量量**：throughput（完成比例 / 时长，RTC）、motion duty cycle（Fast Plans）、completion time + 固定时长内完成数（SmolVLA）、SCT（success × 最短时间/实际时间）。
- 注入时延的曲线有三种形状：**无影响**（RTC 在 +200 ms 内）、**连续下降**（同步执行线性下降；CARLA DS 随 FPS 下降）、**门槛型**（naive async 在 d≥3 步后陡降；streaming perception 在帧间隔附近有 sweet spot）。
- 系统领域的两个现成范式最适合"都可行时排序"：**time-utility function（TUF）**（把完成时刻映射为价值，累加）和 **goodput / capacity**（在 SLO attainment ≥ X% 的约束下可持续的最大速率）。
- 判断：对"executor 硬 200 ms deadline + planner 时延 L 应小于检查间隔 Δ"的异步 planner–executor，建议主指标用 **SCT 型 success-weighted time**（或 throughput），辅指标用 **capacity：在 executor deadline-miss = 0 且 P(L≤Δ) ≥ 99% 下可支持的最小 Δ（或最大检查频率）**。

## 1. 来源表

| # | 来源（发表状态与链接） | 自变量 | 质量指标 | 曲线形状与数字（位置） | 平台 |
|---|---|---|---|---|---|
| 1 | Black, Galliker, Levine. *Real-Time Execution of Action Chunking Flow Policies* (RTC). NeurIPS 2025. [arXiv 2506.07339](https://arxiv.org/abs/2506.07339) | 注入时延 d（控制步）：Kinetix d=0–4（H=8）；真机 +0/+100/+200 ms ≈ d≈6/11/16（Sec 4.2） | Kinetix: success rate；真机："average throughput" = 完成任务比例 / episode 时长（Sec 4.2, Fig 6） | 真机：RTC "completely robust to injected delay, showing no degradation"，synchronous "degrades linearly"；TE 在 +100/+200 ms 触发保护停机无法运行（Sec 4.2）。Kinetix：RTC 最稳，与 BID 的差距随 d 增大而扩大；TE 在 d=0 已差（Sec 4.1, Fig 5，原文无表格数字）。形状：RTC=无影响，sync=连续下降 | π0.5，Δt=20 ms，基础延迟 76 ms（RTC 97 ms），RTX 4090 计时（Table 1）；双臂真机 6 任务、480 episodes |
| 2 | Sendai et al. *Leave No Observation Behind: Real-time Correction for VLA Action Chunks* (A2C2). arXiv 预印本 2025-09。[2509.23224](https://arxiv.org/abs/2509.23224) | 时延 d 与 execution horizon e 的网格（d≤e≤H−d） | success rate（每点 2048 rollouts） | Kinetix e=max(d,1)（Table 4）：Naive 78.1→75.7→59.3→51.2，RTC 88.2→84.1→74.8→62.9，A2C2 89.5→87.4→87.4→86.5（d=1..4）。Naive=门槛型（d≥3 陡降），RTC=连续下降，A2C2≈平。LIBERO-Spatial e=40（Table 7）：Naive 0.74/0.65/0.66/0.67，A2C2 0.83/0.86/0.86/0.84（d=1/3/5/10）；Table 1 与 Table 7 数值不一致 | Kinetix 仿真、LIBERO；RTX 5080 laptop 上 SmolVLA ≈101 ms/step，correction head ≈4.7 ms（Table 8） |
| 3 | Shukor et al. *SmolVLA* (async inference). arXiv 预印本 2025-06 †作者列表未逐一核对。[2506.01844](https://arxiv.org/abs/2506.01844) | 队列阈值 g（|A_t|/n < g 时发新观测）：g=0 顺序 / 0.7 异步 / 1 每步推理；sync vs async | success rate；平均完成时间；固定时长（60 s）内完成次数 | Fig 5：成功率 sync 78.3% vs async 73.3%（"comparable"）；完成时间 13.75 s vs 9.70 s（−30%）；60 s 内 pick-place 9 vs 19 次。理论下界 g ≥ E[ℓ_S]/(nΔt) 保证队列不空 | SO-100 臂；GPU 未写明 |
| 4 | Xie et al. *Fast Plans, Faithful Actions*. arXiv 预印本 2026-09。[2609.30833](https://arxiv.org/abs/2609.30833) | planner 解码方式 → planning latency（Token-AR 1094 ms vs Block-AR 125 ms） | **motion duty cycle** = 执行动作的 wall-clock 时间占比（Table 3 定义）；success rate | Table 3A：round-trip 1266→293 ms；duty cycle 28.7%→65.6%；剩余时延主要是 action expert ≈160 ms。Table 5：成功 48/60（Token-AR）、46/60（Block-AR）、51/60（+NGM）。即时延下降主要换来 duty cycle，成功率基本不变（无影响型） | Rokae AR5-5 双臂；GPU 未写明 |
| 5 | Lim et al. *DualVLN on Jetson Thor: Measuring Inference Cost, Observation Freshness, and Navigation Quality*. NeurIPS 2026 workshop poster（On-Device Intelligence）。[页面](https://neurips.cc/virtual/2026/170076) | 运行时设置 → **observation age**（动作执行时所依据观测的年龄）；planner/S1 时延 | success（12 条路径）、p95 observation age、S1 service rate | 中位时延加速 planning 6.8×、trajectory 3.9×，但成功率 0.556→0.458（移植后）；每个 S1 输出只用一次使 p95 observation age 2.91 s→0.99 s，service rate 相近。作者结论：推理速度本身不决定 freshness 与导航质量（摘要；小规模仿真，计算期间场景静止） | Jetson Thor，TensorRT + NVFP4 planner，H1 人形仿真 |
| 6 | Zhou et al. *NavGPT-3: Harnessing Context in a Hierarchical Navigation Runtime*. arXiv 预印本 2026-10（v2）。[2610.10787](https://arxiv.org/abs/2610.10787) | 运行时结构（sequential/overlapping × queued/priority）→ 事件响应时延 | **trigger-to-halt p50/p95**（速度低于阈值持续一段时间才算 halt）；missed stops；**repair time** | Table 10（Sec 4.8.3）：halt p95 2150/1840/420/1310/180 ms（planner+tools / seq+queued / seq+priority / overlap+queued / overlap+priority）；missed stops 9/7/2/5/0 of 30；repair time 14.2/12.8/11.5/9.6/7.9 s | Go2 四足；VLA 在 Jetson Thor；planner 为托管服务 |
| 7 | Li, Wang, Ramanan. *Towards Streaming Perception*. ECCV 2020 (Oral)。[2005.10420](https://arxiv.org/abs/2005.10420) | 检测器推理时延（模型/尺度/硬件） | **streaming AP**：每个 GT 时刻与当时已可用的最新输出配对（zero-order hold）后计算 AP（Sec 3.1, Eq 3–4） | HTC offline AP 38.0 → streaming 6.2（700.5 ms）；RetinaNet 36.4 ms 仅 5.5；Mask R-CNN 56.7 ms 得 10.6，加调度 13.0（Table 1）；V100 上 HTC 338 ms → 8.2（Table F）。形状：有 sweet spot（既非最快也非最准），且位于 33 ms 帧间隔之上 | GTX 1080 Ti；V100（App A.7）；Argoverse-HD |
| 8 | Ji et al. *Software-Hardware Co-optimization for Modular E2E AV Paradigm*. arXiv 预印本 2026-01。[2601.07393](https://arxiv.org/abs/2601.07393) | 实测推理时延 → CARLA 跳 tick：n=max(0, int(t_i/Δt)−1)，Δt=0.05 s，期间保持上一控制 | Driving Score、comfort、efficiency；复合指标 EER_AV | UniAD 固定 FPS（Table 2）：DS 32.85（1 FPS）→39.53（20 FPS）→≈40.63（24 FPS），20–24 FPS 后饱和（连续上升后平台）；VAD_rt 9.52 FPS 时 DS 31.92，比 42.35 基线低 24.62%。EER_AV = Σ Q_i/220，Q_i = w_DS·x_DS + C·(w_DE·x_DE + w_DC·x_DC + w_E·x_E)，C=碰撞则 0；公式中无显式 latency 项（时延只经 DS 间接进入） | 8×RTX 4090，CARLA 0.9.15，TensorRT |
| 9 | Anderson et al. *On Evaluation of Embodied Navigation Agents*. arXiv 工作组报告 2018。[1807.06757](https://arxiv.org/abs/1807.06757) | —（定义型） | **SPL** = S·L/max(P,L)（公式取自 #10 Eq 1） | 不含时间；SPL 对"更快但路径稍长"不敏感 | — |
| 10 | Yokoyama, Ha, Batra. *Success Weighted by Completion Time (SCT)*. arXiv 2021（v2 2023-10），会议 †（常引为 IROS 2021，未核实）。[2103.08022](https://arxiv.org/abs/2103.08022) | agent 动力学 → 完成时间 | **SCT** = S·T/max(C,T)，C=实际完成时间，T=按动力学可达的最短时间（Eq 2） | Table I：point-turn SCT 65.82 / SPL 88.41 vs unicycle-6 SCT 88.92 / SPL 88.07——SPL 几乎打平，SCT 拉开；真机 Table II：SCT 37.39 vs 56.98（≈74 s vs 48 s），SPL 反而偏向慢者 | Habitat 仿真 + LoCoBot 真机 |
| 11 | Jensen, Locke, Tokuda. *A Time-Driven Scheduling Model for Real-Time Operating Systems*. RTSS 1985, pp.112–122。[researchr](https://researchr.org/publication/JensenLT85/reviews) | 任务完成时刻 | **time-value / time-utility function (TUF)**：每个任务的价值是完成时刻的函数，调度目标为累计价值最大 † | 硬 deadline = 阶跃函数，软 deadline = deadline 后递减 †（原文未能获取，概念据领域通识） | — |
| 12 | Zhong et al. *DistServe*. OSDI 2024。[2401.09670](https://arxiv.org/abs/2401.09670) | 请求到达率 | **goodput** = "the maximum rate that can be served within both TTFT and TPOT constraints"，以 >90% 请求满足 SLO 为 attainment 目标（摘要） | 报告 7.4× 更多请求或 12.6× 更紧 SLO（摘要）。形状：SLO attainment 随 rate 先平后陡降，goodput 取拐点 | GPU 集群 |

BID（Liu et al., [2408.17355](https://arxiv.org/abs/2408.17355)，ICLR 2025 †）本身主要扫 execution horizon（开环/闭环），其时延维度的对比见 RTC Fig 5；BID N=16 时延 115–223 ms（RTC Table 1）。VLASH（[2512.01031](https://arxiv.org/abs/2512.01031)）报告 reaction latency −11.8×、完成速度 1.5–2.0×（摘要），未核实是否有 delay sweep。

## 2. 综合

### 2.1 "都可行时"排序的指标形态

| 形态 | 代表 | 定义 | 对"都满足时限"的分辨力 |
|---|---|---|---|
| A. success × 时间效率 | SCT（#10）、RTC throughput（#1）、SmolVLA 固定时长完成数（#3） | S·T_min/max(C,T_min)；完成比例/时长；N_done/固定时长 | 高：时延 → 停顿/慢动作 → 完成时间，直接反映在分数里 |
| B. 时间利用率 | motion duty cycle（#4） | 执行时间 / wall-clock | 中：只量"没在等"，不量"做得对" |
| C. time-utility function | TUF（#11）、streaming AP（#7，相当于 0 阶保持下的隐式效用） | Σ U_i(t_complete)；或在每个时刻用最新输出评分 | 高，但需要选 U 的形状 |
| D. capacity / goodput | DistServe goodput（#12） | 在 SLO attainment ≥ X% 约束下最大可持续速率 | 高：所有系统都可行时比"余量"；与 tail 直接挂钩 |
| E. tail 响应时延 + 失误计数 | NavGPT-3 halt p95 + missed stops（#6）；p95 observation age（#5） | 事件→物理效果的 p95；超时计数 | 适合安全事件，不是单一质量分 |
| F. 质量–时延曲线下面积 | 未见论文显式使用；RTC/A2C2 的 sweep 天然支持 | ∫ Q(d) dd / d_max | 判断：适合比较"对时延的鲁棒性"，不适合比较单一部署点 |

### 2.2 常见图形

- quality vs 注入时延 d 曲线（RTC Fig 5、A2C2 Fig 5–6）；多方法同轴，看斜率与拐点。
- 累计任务进度 vs 时间（RTC Fig 6，分"去掉停顿的控制步"和"wall-clock"两版），用来把 throughput 拆成"动作质量"与"等待"。
- 执行时间线 / duty cycle 甘特图（Fast Plans Fig 4a）。
- 精度–时延散点 + sweet spot（Streaming Perception）。
- 分位数表（p50/p95）与失败计数（NavGPT-3 Table 10；DualVLN p95 observation age）。
- SLO attainment vs request rate，goodput 取 90% 交点（DistServe 类）。
- 带 deadline 竖线的时延 CDF、margin 直方图：判断：本次调研中未在上述机器人论文里看到作为主图，实时系统文献中常见 †。

### 2.3 判断：异步 planner–executor（executor 硬 200 ms deadline，planner 时延 L 应 < 检查间隔 Δ）

- executor 的 200 ms 是**硬约束**：按 TUF 观点是阶跃效用，满足即可、不应进入加权分数；只作可行性门槛（deadline-miss 计数 = 0，附 p99/max 与 margin 直方图）。
- planner 的 L<Δ 是**软约束**：L 超出 Δ 时，后果是 observation age 增大、决策延后，而非立即失败；质量随 L/Δ 呈门槛型（类似 A2C2 naive 在 d≥3 的拐点）。
- 推荐指标 1（主，任务质量）：**SCT 型 success-weighted completion time**（或 RTC 式 throughput = 完成比例/时长）。它同时吸收"planner 慢 → 执行停顿/走错后修复"的代价，且所有可行系统都能被区分。
- 推荐指标 2（辅，系统余量）：**capacity at quality target**——在 executor deadline-miss = 0 且 P(L ≤ Δ) ≥ 99% 的约束下，可支持的最小 Δ（即最大检查频率 1/Δ），或在固定 Δ 下可共置的最大负载。这是 goodput 的直接翻译，回答"还能收紧多少"。
- 配套图：L 的 CDF 带 Δ 竖线 + executor 时延 CDF 带 200 ms 竖线；SCT vs 注入 planner 时延（L/Δ = 0.25…2）曲线，用于确认拐点在 L≈Δ 附近。
