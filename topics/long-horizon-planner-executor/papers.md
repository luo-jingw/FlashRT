# long-horizon-planner-executor 论文目录

- 快照日期：2026-11-09
- 年份：首次公开的年份（arXiv 或正式发表，取较早者）。
- 会议/期刊：取自 arXiv comment 字段或论文集；"预印本"表示没有找到发表处。标 † 的条目有未核实项，见根目录 `issues.md` → ISSUE-113。
- 标题、年份、arXiv 编号都在 arxiv.org 页面（或会议页面）上核对过；数字取自摘要或正文，并注明位置。只在图中给出、正文没有数值的，写"仅图"。
- 分节对应 [survey.md](survey.md) 的第 2–5 节。

## A. 动机：长时程任务与分层

### A1. 两模型分层（独立的 VLM 规划器 + VLA 执行器）

- **Hi-VLA 编排研究** · 2026 · 预印本 · [arXiv 2606.10267](https://arxiv.org/abs/2606.10267)
  - What Matters in Orchestrating Robot Policies: A Systematic Study of Hierarchical VLA Agents
  - 同一 VLA、同一任务提示下比较三种设置（表 1）：长时程 flat 25.30 / 简单分层 40.56 / 最佳分层 67.08；短时程 69.63 / 69.57 / 78.22；真机 ALOHA 3/15、9/15、12/15
- **VoLo** · 2026 · 预印本 · [arXiv 2606.07723](https://arxiv.org/abs/2606.07723)
  - VoLo: A Physical Orchestrator for Open-Vocabulary Long-Horizon Manipulation
  - 同一个 π0.5 加 VLM 调度：跨套件成功率 12.57 → 41.80（表 2）；规划器换成 Qwen3-VL-8B 只有 19.95，"with weaker VLMs the gain becomes marginal"（§5.4）
- **Gemini Robotics 1.5** · 2025 · 预印本 · [arXiv 2510.03342](https://arxiv.org/abs/2510.03342)
  - Gemini Robotics 1.5: Pushing the Frontier of Generalist Robots with Advanced Embodied Reasoning, Thinking, and Motion Transfer
  - 同一 VLA：单独思考进度分最高 44%，加 ER 1.5 编排器常接近 80%（§5）；规划类失败占评测次数：编排器为 Gemini 2.5 Flash 时 25.5%，为 ER 1.5 时 9%（表 1，两种编排器之间的比较）
- **Hi Robot** · 2025 · 预印本 · [arXiv 2502.19417](https://arxiv.org/abs/2502.19417)
  - Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models
  - 同一 π0 低层；对 GPT-4o 高层指令准确率高 40% 以上（§5.3）；对 flat π0 的数值仅图；作者称两层分开 "not fundamental"（§6）
- **RoboCerebra** · 2025 · 预印本 · [arXiv 2506.06677](https://arxiv.org/abs/2506.06677)
  - RoboCerebra: A Large-scale Benchmark for Long-horizon Robotic Manipulation Evaluation
  - 平均 9.1 个子步、2,972 个仿真步；OpenVLA 4.57 → 加规划器 16.04（表 3），真值计划 25.16（表 4）†
- **LoHo-Manip** · 2026 · 预印本 · [arXiv 2604.21924](https://arxiv.org/abs/2604.21924)
  - Long-Horizon Manipulation via Trace-Conditioned VLA Planning
  - VLABench π0.5 0.24 → 0.39（表 4）；episode 86 s，不加 manager 72 s（表 8）
- **HiVLA** · 2026 · 预印本 · [arXiv 2604.14125](https://arxiv.org/abs/2604.14125)
  - HiVLA: A Visual-Grounded-Centric Hierarchical Embodied Manipulation System
  - RoboTwin 2.0 Hard 平均：π0.5 36.4%，HiVLA 73.2%（表 1）；基线都在作者生成的数据上微调
- **Agentic Robot** · 2025 · 预印本 · [arXiv 2505.23450](https://arxiv.org/abs/2505.23450)
  - Agentic Robot: A Brain-Inspired Framework for Vision-Language-Action Models in Embodied Agents
  - LIBERO-Long 61.6，OpenVLA 53.7（表 1）；远低于 flat π0.5 公布的 92.4
- **MemER** · 2025 · 预印本 · [arXiv 2510.20328](https://arxiv.org/abs/2510.20328)
  - MemER: Scaling Up Memory for Robot Control via Experience Retrieval
  - 分层系统内部比较记忆方式（表 1），没有 flat 基线
- **Tri-System π0.5** · 2026 · 预印本 · [arXiv 2603.05185](https://arxiv.org/abs/2603.05185)
  - Critic in the Loop: A Tri-System VLA Framework for Robust Long-Horizon Manipulation
  - Arrange Tableware：单系统 0/10、双系统 6/10、三系统 9/10（表 1）†
- **Fast Plans** · 2026 · 预印本 · [arXiv 2609.30833](https://arxiv.org/abs/2609.30833)
  - Fast Plans, Faithful Actions: Closing the Planning–Execution Gap in Hierarchical Vision-Language-Action Models
  - LIBERO-Long 96.2 对 π0.5 公布值 92.4，另两个变体 92.2 / 91.0（表 2）；motion duty cycle 28.7% → 65.6%（表 3A）
- **HAMSTER** · 2025 · 预印本 · [arXiv 2502.05485](https://arxiv.org/abs/2502.05485)
  - HAMSTER: Hierarchical Action Models For Open-World Robot Manipulation
  - 真机 7 个泛化维度上比 OpenVLA 平均高 20%（摘要）；泛化证据，不是长时程证据

### A2. 单模型（同一模型内做高层推理）与反证

- **π0.5** · 2025 · 预印本 · [arXiv 2504.16054](https://arxiv.org/abs/2504.16054)
  - π0.5: a Vision-Language-Action Model with Open-World Generalization
  - 同一模型先预测子任务再出动作；优于人类高层 oracle，GPT-4 零样本高层最差（§V-E，图 13 仅图）
- **OneTwoVLA** · 2025 · 预印本 · [arXiv 2505.11917](https://arxiv.org/abs/2505.11917)
  - OneTwoVLA: A Unified Vision-Language-Action Model with Adaptive Reasoning
  - 平均 87%，比 Gemini 2.5 Pro + π0 双系统高 24%（§4.1，图 6）；双系统的规划器未针对该机器人微调
- **LoHoVLA** · 2025 · 预印本 · [arXiv 2506.00411](https://arxiv.org/abs/2506.00411)
  - LoHoVLA: A Unified Vision-Language-Action Model for Long-Horizon Embodied Tasks
  - LoHoSet 上单模型优于早期分层方案 LoHoRavens（表 2），两者执行器与数据不同
- **Long-VLA** · 2025 · CoRL 2025 · [arXiv 2508.19958](https://arxiv.org/abs/2508.19958)
  - Long-VLA: Unleashing Long-Horizon Capability of Vision Language Action Model for Robot Manipulation
  - 端到端单模型，phase-aware input masking；L-CALVIN 数值未核实 †
- **MemoryVLA** · 2025 · ICLR 2026 · [arXiv 2508.19236](https://arxiv.org/abs/2508.19236)
  - MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation
  - 单模型带记忆库；长时程任务比最佳基线高 26 点（摘要），没有分层基线
- **BEHAVIOR 2025 第 1 名** · 2025 · 预印本 · [arXiv 2512.06951](https://arxiv.org/abs/2512.06951)
  - Task adaptation of Vision-Language-Action model: 1st Place Solution for the 2025 BEHAVIOR Challenge
  - 单模型（π0.5 改造，内部阶段预测）；任务平均 6.6 min；private 成功率 0.1240（表 1）；第 2 名 Openpi Comet（[arXiv 2512.10071](https://arxiv.org/abs/2512.10071)）也是单模型

### A3. 受控范式比较

- **VLA-OS** · 2025 · 预印本 · [arXiv 2506.17561](https://arxiv.org/abs/2506.17561)
  - VLA-OS: Structuring and Dissecting Planning Representations and Paradigms in Vision-Language-Action Models
  - Integrated 与 Hierarchical 都优于 ActionOnly，两者 "largely comparable"；Hierarchical 泛化最好，但训练与推理更慢（项目页 Findings）†

### A4. 基准与长时程衰减参照

- **GR-1** · 2023 · 预印本 · [arXiv 2312.13139](https://arxiv.org/abs/2312.13139)
  - Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation
  - CALVIN ABC→D 连续 1→5 个任务：0.854 → 0.401；HULC 0.418 → 0.011（表 1）

## B. 规划器延迟与不同步的代价

### B1. 秒级慢路径的延迟与任务结果

- **Leap-VLA** · 2026 · 预印本 · [arXiv 2611.01011](https://arxiv.org/abs/2611.01011)
  - Leap-VLA: High-Rate Robot Control from a Low-Rate VLM Backbone
  - LIBERO-Spatial，同频训练：刷新周期 0.2 / 1 / 2 s 下 97.5 / 82.9 / 61.4；按部署时钟训练 1 / 2 s 下 95.9 / 95.5（表 III）
- **Slow Brain, Fast Planner** · 2026 · 预印本 · [arXiv 2606.20458](https://arxiv.org/abs/2606.20458)
  - Slow Brain, Fast Planner: Latency-Resilient VLM-Augmented Urban Navigation
  - 注入 0–5 s VLM 延迟：直接沿用的方式 2 s 后崩溃，融合方式到 5 s 仍高于 80%（图 6，仅图）；真机 takeovers/100 m 0.87 对 8.15（表 3）
- **NavGPT-3** · 2026 · 预印本 · [arXiv 2610.10787](https://arxiv.org/abs/2610.10787)
  - NavGPT-3: Harnessing Context in a Hierarchical Navigation Runtime
  - 真机 Go2（板载 VLA 在 Jetson Thor 上）：规划器与 VLA 重叠执行 repair time 7.9 s、顺序执行 12.8 s，stale command 1/30 对 8/30（表 10，各 30 次）；无周期检查时中位 51 步才拿回控制（表 5）
- **DualVLN on Thor** · 2026 · NeurIPS 2026 Workshop · [OpenReview Ei1Wgdo2I0](https://openreview.net/forum?id=Ei1Wgdo2I0)
  - DualVLN on Jetson Thor: Measuring Inference Cost, Observation Freshness, and Navigation Quality
  - 单块 Thor；p95 观测年龄 2.91 → 0.99 s，成功率 0.556 → 0.458（摘要）；全文未拿到 †
- **VLALagBench** · 2026 · 预印本 · [arXiv 2611.03103](https://arxiv.org/abs/2611.03103)
  - VLALagBench: Understanding When and Why VLA Inference Latency Matters
  - 定义 observation staleness 与 idle steps（§3.5）；π0.5 sync 100 ms：移动目标 set-down 71.5% → 40.5%（§4.2）
- **AsyncVLA** · 2026 · 预印本 · [arXiv 2602.13476](https://arxiv.org/abs/2602.13476)
  - AsyncVLA: An Asynchronous VLA for Fast and Robust Navigation on the Edge
  - 远端 4090 + Jetson Orin 适配器，WiFi 延迟 0.28–6 s；成功率 0.85 对 OmniVLA 0.45（表 I）
- **CloudEdgeVLA** · 2026 · 预印本 · [arXiv 2608.00569](https://arxiv.org/abs/2608.00569)
  - Latency-Tolerant Cloud-Edge Collaborative Vision-Language-Action Models via Emergent Representational Specialization
  - delay-retention：d_max = 40 时平均保留 73.5%，VLASH 约 1.7%（表 1）

### B2. 同步与异步：慢路径计算期间执行器在做什么

- **StreamingVLA** · 2026 · 预印本 · [arXiv 2603.28565](https://arxiv.org/abs/2603.28565)
  - StreamingVLA: Streaming Vision-Language-Action Model with Action Flow Matching and Adaptive Early Observation
  - 相邻执行间的空闲 T_halt 232.3 → 36.0 ms，成功率 94.9 对 95.1%（表 1）
- **SmolVLA** · 2025 · 预印本 · [arXiv 2506.01844](https://arxiv.org/abs/2506.01844)
  - SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics †
  - 异步推理完成时间 13.75 → 9.70 s，"about 30% faster"（§4.6，图 5）
- **RTC** · 2025 · 预印本 · [arXiv 2506.07339](https://arxiv.org/abs/2506.07339)
  - Real-Time Execution of Action Chunking Flow Policies
  - 执行器 deadline d = ⌊δ/Δt⌋，d ≤ s ≤ H − d（§2）；指标 average task throughput
- **A2C2** · 2025 · 预印本 · [arXiv 2509.23224](https://arxiv.org/abs/2509.23224)
  - Leave No Observation Behind: Real-time Correction for VLA Action Chunks
  - 比 RTC Kinetix +23 点、LIBERO-Spatial +7 点（摘要）
- **EagleVLA** · 2026 · 预印本 · [arXiv 2607.12659](https://arxiv.org/abs/2607.12659)
  - EagleVLA: Onboard Real-Time Robot Control via Foresight-Aligned Asynchronous Inference
  - reaction time（环境变化到机器人响应）在 Thor 上 457.9 → 61.1 ms（表 4）
- **OxyGen** · 2026 · 预印本 · [arXiv 2603.14371](https://arxiv.org/abs/2603.14371)
  - OxyGen: Unified KV Cache Management for VLA Inference under Multi-Task Parallelism
  - 指标 action frequency = H / 帧延迟；Thor 约 27 Hz；没有统计 deadline miss †

- **Understanding Asynchronous Inference** · 2026 · 预印本（学士论文）· [arXiv 2605.08168](https://arxiv.org/abs/2605.08168)
  - Understanding Asynchronous Inference Methods for VLA
  - LIBERO、d = 20：naive 约 10–12%，A2C2 约 58%；作者承认 idle 没有测量、延迟按固定整数处理（§7）†

### B3. 驾驶中的快慢系统

- **Think at 5 Hz, Act at 20 Hz** · 2026 · 预印本 · [arXiv 2607.15621](https://arxiv.org/abs/2607.15621) †
  - 慢路径每 4 个 tick 刷新一次；没有闭环 staleness 消融
- **AsyncDriver** · 2024 · 预印本 · [arXiv 2406.14556](https://arxiv.org/abs/2406.14556)
  - Asynchronous Large Language Model Enhanced Planner for Autonomous Driving †
  - 每 3 帧调用一次 LLM：推理时间约 −40%，精度约 −1%

### B4. 事件触发与周期调用

- **VLA-Corrector** · 2026 · 预印本 · [arXiv 2607.01804](https://arxiv.org/abs/2607.01804)
  - 失配连续 5 步超过阈值才重规划：6.88 对 9.41 次调用 / episode，成功率 73.1% 对 69.4%（表 2）†

- **LiteVLA-H** · 2026 · 预印本 · [arXiv 2605.00884](https://arxiv.org/abs/2605.00884)
  - 同一个 256M 模型的动作与语义两路，Orin 上 50.65 / 149.90 ms（表 2）；事件触发为可选功能，未评测 †

### B5. 评测框架

- **vla-eval** · 2026 · 预印本 · [arXiv 2603.13966](https://arxiv.org/abs/2603.13966)
  - 阻塞式 predict，"Supported metrics are limited to task success rate"（Limitations）
- **BEHAVIOR Challenge 2025 评测规则**（官方页面）
  - 次级指标 simulated time "hardware-independent"，推理延迟不计入任务时间

## C. 边缘 SoC 上的系统现象

### C1. 统一内存的带宽争用

- **Ripple** · 2026 · 预印本 · [arXiv 2611.01067](https://arxiv.org/abs/2611.01067)
  - Ripple: Deadline-Aware CPU–GPU Sharing for Embodied Inference
  - DGX Spark（GB10，LPDDR5X 273 GB/s）：LLM decode 使 CPU 任务慢 1.7–2.5×，H200 上 "no impact"；LLM 从 48 个 SM 缩到 8 个 "very limited effect"（§2.2，图 3a）；k = 1.5 时超时 0.78%、7.00 tok/s（表 7）
- **EdgePace** · 2026 · 预印本 · [arXiv 2610.37307](https://arxiv.org/abs/2610.37307)
  - EdgePace: Deadline- and Freshness-Aware LLM Scheduling on Unified-Memory Platforms
  - Orin Nano：访存型背景使相机任务 p95 1.52×、超时 72.78%，算力型背景影响可以忽略（§1，图 1）；带宽阈值告警 93.84% 是误报（§2.4）
- **UMA 内存共享表征** · 2026 · EuroSys 2027 · [arXiv 2610.13125](https://arxiv.org/abs/2610.13125)
  - Characterizing CPU-GPU Memory Sharing over Unified Memory Architecture
  - Orin 上 L2 优化的 GEMM 在 uncached 内存上 −16%（§4.2.2）
- **HaX-CoNN** · 2024 · PPoPP 2024 · [arXiv 2308.05869](https://arxiv.org/abs/2308.05869)
  - Shared Memory-contention-aware Concurrent DNN Execution for Diversely Heterogeneous System-on-Chips
  - GPU 与 DLA 争用 DRAM；不考虑争用的决策最多降 70%（§2，§5.2）
- **iGPU L2 锁定** · 2023 · WCET 2023 · [DOI 10.4230/OASIcs.WCET.2023.3](https://doi.org/10.4230/OASIcs.WCET.2023.3)
  - Exploring iGPU Memory Interference Response to L2 Cache Locking
  - AGX Orin 上 LPDDR5 干扰约占卷积 WCET 的 26%（§4.3）
- **XSched** · 2025 · OSDI 2025 · [USENIX](https://www.usenix.org/conference/osdi25/presentation/shen-weihang)
  - XSched: Preemptive Scheduling for Diverse XPUs
  - Orin 上 GPU 与 NPU "may compete for the memory bandwidth and power supply"（§7.2）；前台 P99 1.67× / 1.55× → 1.18× / 1.09×（图 10）
- **KerColle** · 2026 · PACT 2026 · [arXiv 2609.22335](https://arxiv.org/abs/2609.22335)
  - KerColle: Unlocking Fine-Grained GPU Concurrency in Vision-Language-Action Models
  - 模拟 4090 / 5090：动作头与 VLM 并发时出块速率约减半，原因是 CTA 分发器队头阻塞（§2.4）

### C2. DVFS 与内存时钟（EMC）

- **Edge-Inference Governors** · 2026 · 预印本 · [arXiv 2606.16106](https://arxiv.org/abs/2606.16106)
  - Edge-Inference Governors Need Memory-Clock State
  - Orin NX / Nano：EMC-blind 拟合超时 25–28%，EMC-aware ≤ 0.9%（摘要）；超时成簇；明确排除 HBM 服务器，因其内存时钟 "not a deployment-time knob"（§XII）
- **Arya & Simmhan** · 2025 · PAISE 2025 · [arXiv 2506.09554](https://arxiv.org/abs/2506.09554)
  - Understanding the Performance and Power of LLM Inferencing on Edge Accelerators
  - AGX Orin：EMC 3200 → 665 MHz，LLM 延迟约 +370%（§3.4）
- **DVFSLM** · 2026 · 预印本 · [arXiv 2609.13153](https://arxiv.org/abs/2609.13153)
  - DVFS for Small Language Model Inference on Mobile Edge Devices
  - Orin 上 EMC 0.204 GHz 时 "severe QoS degradation"（§5.3）
- **Hydra** · 2026 · IISWC 2026 · [arXiv 2608.25053](https://arxiv.org/abs/2608.25053)
  - Hydra: Phase-Aware Workload Characterization of LLM Inference across Edge SoC Generations, Backends, and Quantization Levels
  - Xavier / Orin / Thor 的 LLM 分阶段刻画；Thor 7–8B Q4 decode 35–44 tok/s（§V-A）；没有 Thor 带宽与 warm-up 数据
- **StarSD** · 2026 · 预印本 · [arXiv 2601.21622](https://arxiv.org/abs/2601.21622)
  - 独显上空闲间隙后的每 token 时间变慢在几十毫秒量级，锁频后仍存在（§5.3，图 6）

- **边缘隔离实测** · 2026 · 预印本 · [arXiv 2601.07600](https://arxiv.org/abs/2601.07600)
  - A100 上比较 MPS 与 MIG，Orin 上比较 MPS 与 green context：green context 没有内存隔离；Orin Nano 两模型并行时 GPU 频率约减半 †

### C3. 数据中心对照

- **Host-CPU 共置干扰** · 2026 · 预印本 · [arXiv 2609.05425](https://arxiv.org/abs/2609.05425)
  - Measurement-Driven Diagnosis and Mitigation of Host-CPU Co-location Interference in Single-GPU LLM Serving on a Multi-GPU Server
  - 与 nginx 共置吞吐 −78.8%，但 "the main observed problem is not slower GPU kernels"（摘要）
- **Orion** · 2024 · EuroSys 2024 · [DOI 10.1145/3627703.3629578](https://doi.org/10.1145/3627703.3629578)
  - Orion: Interference-aware, Fine-grained GPU Sharing for ML Applications †
  - 只在资源画像互补时共跑 kernel；把 memory-bound decode 与计算密集负载共置列为后续工作（§7）

## D. 刻画类论文的写法参照

- **VLA across XPUs** · 2026 · 预印本 · [arXiv 2604.24447](https://arxiv.org/abs/2604.24447)
  - Characterizing Vision-Language-Action Models across XPUs: Constraints and Acceleration for On-Robot Deployment
  - 6 条 "Finding #N"，每条一句结论配一张图或表；含 4090 / Orin / Thor 的两阶段不均衡与 roofline。精读笔记：[notes/2026-vla-xpu-characterization.md](notes/2026-vla-xpu-characterization.md)
