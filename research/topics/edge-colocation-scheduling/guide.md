# 导读：系统论文的思路，以及本方向的现状

本页是入口，约十分钟读完。其余文件是查证用的资料。

## 1. 一篇系统论文的贡献是一条链

隐含假设 → 量化损失 → 洞察 → 机制 → 对齐的评估

| 步骤 | [Clockwork](https://www.usenix.org/conference/osdi20/presentation/gujarati)（OSDI'20） | [vLLM](https://doi.org/10.1145/3600006.3613165)（SOSP'23） |
|---|---|---|
| 隐含假设 | 推理时延不可预测，所以只能并发、事后补救 | KV cache 必须按最大长度预留连续内存 |
| 量化损失 | 并发时吞吐最多 +25 %，尾时延却升约 100 倍 | 只有 20.4–38.2 % 的 KV 内存真正存了 token |
| 洞察（X 成立，所以 Y） | DNN 推理时延是确定的，所以可以提前排好顺序 | KV cache 的困境和进程内存同构，所以可以分页 |
| 机制 | 集中做决策，每块 GPU 一次只跑一个推理 | PagedAttention |
| 评估 | SLO 收紧时 goodput 仍稳住 | 内存利用率和吞吐 |

novelty 几乎都在"洞察"这一步：发现一个现有系统没有利用的性质。机制是这个性质的直接推论。

## 2. 洞察通常从哪里来

最常见的四种：

1. **某个量其实可预测或可负担**：Clockwork 的时延可预测；[REEF](https://www.usenix.org/conference/osdi22/presentation/han) 发现 kernel 大多幂等，所以可以杀掉重跑。
2. **换调度粒度**：[Orca](https://www.usenix.org/conference/osdi22/presentation/yu) 从按请求改成按迭代；[Hummingbird](https://arxiv.org/abs/2601.04071) 从按 kernel 改成按一波 block。
3. **打破公认的权衡**：[IX](https://www.usenix.org/conference/osdi14/technical-sessions/presentation/belay) 同时拿到吞吐、时延和隔离；Hummingbird 同时守住 SLO 和利用率。
4. **重新定义指标**：[DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) 定义 goodput，即满足 SLO 的最大请求率。

完整的 8 种模式见 [survey.md](survey.md) §1.2。

## 3. 好的评估回答三个问题

1. 洞察的前提真的成立吗？要有一个单独的测量，例如 Clockwork 测出 p99.99 与中位数只差 0.03 %。
2. 最可能的质疑是什么？是否单独测过？例如 DistServe 测出拆分后的传输只占总延迟 0.1 % 以下。
3. 放弃了什么？在什么条件下不成立？例如 FlashAttention 多算了一些 FLOPs，换来 HBM 读写量减少约 9 倍。

## 4. 本方向的文献走到了哪里

- **已有的做法**：
  - GPU 不能打断正在运行的 kernel，所以都靠控制下发、把不可打断的单元切小来抢占（XSched、Hummingbird）。
  - LLM 服务在同一块卡内按 SM 拆开 prefill 和 decode（MuxWise）。
  - 但这些工作几乎都在独立显卡上做；受保护的一方要么是同一个模型的 decode，要么让步方是没有时延要求的离线任务。
- **边缘 SoC 上**：
  - 多篇工作测到"把 SM 隔开以后，带宽干扰依然存在"，但没有一篇提出隔离带宽的机制。
  - 在 Thor 上测过两模型共置干扰的只有 [PR #193](https://github.com/NVIDIA/TensorRT-Edge-LLM/pull/193)：规划器 7B，prefill 约 75 ms。
- **没找到的组合**：
  - 一块集成 GPU 上，同时有截止期严格的小模型和 prefill 达秒级、切小会改变输出的 LLM；
  - 对 MPS 客户端优先级的评估；
  - CPU 与两个 GPU 负载三方争用带宽的量化。

## 5. 拿这条链检查我们自己

先不预设答案，逐条回答：

1. 现有做法（MPS 客户端优先级、静态分 SM）隐含了什么假设？
2. 能不能画一张"实际值对理想值"的图？理想值是什么，例如执行器独占时 37.5 ms，规划器独占时的 L。
3. 有没有一个别人没利用的性质 X？它要能单独测出来。
4. 机制能不能用 X 命名？
5. 评估的主指标是不是就是第 2 步里那个量？

## 6. 先读这三篇

1. [Clockwork](https://www.usenix.org/conference/osdi20/presentation/gujarati)：这条链最干净的例子，原文好读。
2. [Hummingbird 笔记](notes/2026-hummingbird.md)：与本场景最接近的抢占机制，以及它的前提哪些不成立。
3. [Serafini 第 6 章笔记](notes/2026-serafini-thesis-ch6.md)：Thor 上带宽和 SM 数的关系，属于硬件事实。

## 7. 主线与核心问题

- **设定**：异步协议是主战场，同步协议作基线。执行器 200 ms 截止期是门槛（余量 > 0 即可）；规划器的决策延迟 L 要小于检查间隔 Δ。
- **核心问题**：异步下，规划器的决策延迟 L 怎样影响任务质量（SR、完成时间）？文献里没有人在固定调用周期下单独改变 L 并测 SR（[harnesses.md](harnesses.md)）。最接近的证据是 NavGPT-3（规划器与 VLA 重叠并配优先级中断后，真机 SR 73.3 → 83.3）和 Fast Plans（规划提速 8.7 倍，机器人在动的时间占比 29 % → 66 %，SR 基本不变）。
- **答案决定排序指标**（[latency_quality_metrics.md](latency_quality_metrics.md)）：
  - 门槛型（L < Δ 时质量不变）：用能力排序，即两个余量都 > 0 时能支持的最小 Δ、最长上下文、最大 CPU 负载，类比 goodput；
  - 连续型（L 越小越好）：用按完成时间加权的成功率（SCT 类）排序。
- **要画的图**：质量随注入延迟的变化曲线；规划器余量的直方图，标出 Δ。前提是执行器会跟随指令。
- **上下文何时变长**（[long_horizon_conditions.md](long_horizon_conditions.md)）：需要记住看不见的状态（分钟级）、多轮 agent 对话保留工具返回、周期性重读累积的历史、长路径的部分可观测导航、重规划使调用次数长尾。多数已发表系统刻意把上下文控制在 1–12k token；只有 NavGPT-3 一直增长，单次调用从约 8k 到约 30k。
- **硬件不只 Thor**（[unified_memory_socs.md](unified_memory_socs.md)）：DGX Spark 与 Thor 同为 128 GB、273 GB/s；Apple M 系列、AMD Strix Halo 也是统一内存。有官方 GPU 分区的只有 Thor（MIG，论坛称 JetPack 7.2 有 6 / 12 / 20 SM 档，待核实）和 MI300A，两者都不分内存带宽。

## 需要查证时

| 想查什么 | 去哪里 |
|---|---|
| 8 种模式，以及 30 篇各自的一句话洞察 | [survey.md](survey.md) §1–2 |
| 某篇相关论文的原句、贡献、证据 | [decomposition.md](decomposition.md)，[papers.md](papers.md) |
| 经典论文的完整拆解 | [classics_general.md](classics_general.md)，[classics_mlsys.md](classics_mlsys.md) |
| 精读 | `notes/` |
| long horizon 的条件、harness、延迟与质量的指标、统一内存硬件 | [long_horizon_conditions.md](long_horizon_conditions.md)，[harnesses.md](harnesses.md)，[latency_quality_metrics.md](latency_quality_metrics.md)，[unified_memory_socs.md](unified_memory_socs.md) |
