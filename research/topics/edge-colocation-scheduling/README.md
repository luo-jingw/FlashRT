# edge-colocation-scheduling

一块边缘 SoC 上，一个周期性、截止期严格的小模型（执行器）与一个 LLM（规划器）同时运行时的系统文献综述。两个核心部分：
1. 系统论文怎样提出核心贡献：从 20 篇经典或高分系统论文和三篇写作与评审指南中归纳出论证链、8 种论证模式和评估闭环方式；
2. 相关论文按核心要点拆解：30 篇逐篇拆成问题、观察、洞察、贡献、机制、证据、对本场景的限制，并逐条对照它们的前提与 Jetson AGX Thor 上规划器 + 执行器场景的差别。

- 状态：第一轮综述完成（快照 2026-11-17）：略读 30 篇，精读 5 篇；第二轮：经典与高分系统论文 20 篇，30 篇按核心要点拆解
- 最后更新：2026-11-17

## 范围

包含：
- A. GPU 共享与抢占：延迟关键任务和可让步任务共用一块 GPU；
- B. LLM 服务的阶段调度：prefill / decode 干扰，同一 GPU 内划分 SM；
- C. 边缘 SoC 上的多模型实时推理与内存带宽争用（Jetson，统一内存）；
- D. 机器人双系统（慢规划器 + 快执行器）部署的系统面。

不包含：
- 长时程任务为什么用规划器 + 执行器，以及规划器延迟的任务代价，见 [long-horizon-planner-executor](../long-horizon-planner-executor/README.md)；
- 本组在 Thor 和 RTX 5090 上的共置实测，见 [long-horizon-planner-executor](../long-horizon-planner-executor/README.md) 与 [wam-inference-acceleration](../wam-inference-acceleration/README.md)。

## 关键词

GPU preemption, spatial / temporal GPU sharing, MPS, green context, TPC mask, co-location interference, prefill / decode disaggregation, chunked prefill, memory bandwidth contention, unified memory, Jetson Orin, Jetson AGX Thor, real-time DNN scheduling, dual-system VLA

## 文件

| 文件 | 内容 |
|---|---|
| [survey.md](survey.md) | 范围与方法、场景参照、系统论文怎样提出核心贡献（论证链、8 种模式、评估闭环、5 篇精读的对应）、30 篇按核心要点拆解的总表、分类框架（30 篇按受保护方 / 控制维度 / 硬件 / 时限形式归类）、四类脉络、跨类综合（问题表述模板、指标、机制谱系与在 Thor 上的条件、带宽干扰测量、不可抢占单元粒度、前提对照）、未覆盖的组合、阅读顺序、待核实 |
| [classics_general.md](classics_general.md) | 通用系统经典论文 10 篇（MapReduce、GFS、Dynamo、Spanner、Raft、Arrakis、IX、Shenango、Wasted Cores、Borg）与三篇写作和评审指南：论点、观察、洞察、贡献类型、评估如何对应论点，以及归纳的模式 |
| [classics_mlsys.md](classics_mlsys.md) | ML 与 GPU 系统高分论文 10 篇（vLLM、Orca、Clockwork、AlpaServe、PipeDream、Gandiva、FlashAttention、DistServe、Pollux、Mooncake）：同样的拆解与归纳的模式 |
| [decomposition.md](decomposition.md) | 30 篇相关论文逐篇拆解：问题、观察、洞察（原句与位置）、贡献、机制、证据、对本场景的限制 |
| [papers.md](papers.md) | 30 篇略读卡片：场景、目标与指标、核心观察、challenge、机制、基线与评估、与本场景的关系 |
| [search_log.md](search_log.md) | 每类的检索词、来源、元数据核对方式、候选清单与取舍理由 |
| `notes/` | 精读笔记：[XSched](notes/2025-xsched.md)、[Hummingbird](notes/2026-hummingbird.md)、[MuxWise（Drift）](notes/2025-drift.md)、[Serafini 博士论文第 6 章](notes/2026-serafini-thesis-ch6.md)、[OxyGen](notes/2026-oxygen.md) |
