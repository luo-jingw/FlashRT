# long-horizon-planner-executor

长时程机器人任务中的"规划器 + 执行器"双模型系统：为什么长时程任务要用两个模型，两模型放在同一块 Jetson AGX Thor 上时有哪些普通 LLM 服务没有的挑战，用什么目标指标衡量，以及这些挑战可能怎么解决。

- 状态：第一轮调研进行中
- 最后更新：2026-11-09

## 范围

包含：
- 场景动机：长时程任务的定义与基准；分层（规划器 + 执行器）与不分层方案在长时程任务上的公开对比，以及反例；
- 规划器延迟与两模型不同步的代价：机器人空等、决策滞后、过时的计划，以及衡量它们的指标；
- 边缘 SoC 上的系统现象：统一内存的带宽争用、内存与时钟的冷热状态、功耗预算，以及数据中心 LLM 服务为什么没有这些问题；
- 相关系统论文从观察到挑战、机制、评估的写法；
- 本方向的小实验与发现（图表）。

不包含：
- ImageWAM 与规划器本身的推理加速实验、RTX 5090 上的共置实验，见 [wam-inference-acceleration](../wam-inference-acceleration/README.md)；
- 双模型系统的工作负载参数取值（调用节奏、上下文长度、截止期），见 [dual_model_workload_conventions.md](../wam-inference-acceleration/reports/dual_model_workload_conventions.md)。

## 关键词

long-horizon manipulation, hierarchical VLA, planner–executor, dual-system, asynchronous inference, planner latency, reaction time, observation age, Jetson AGX Thor, unified memory, memory bandwidth contention, DVFS, co-location

## 文件

| 文件 | 内容 |
|---|---|
| `survey.md` | 综述（第一轮进行中） |
| `papers.md` | 论文目录（第一轮进行中） |
| `notes/` | 精读笔记（尚无） |
| `reports/` | 汇报材料（尚无） |

## 相关记录

- Thor 实测：根目录 `THOR_STATUS_SUMMARY.md`（`1104_h3`、`1104_h3b`、`1104_h3c`、`1105_h6h4` 轮）
- 空闲后解码变慢：根目录 `issues.md` → ISSUE-112
- RTX 5090 共置报告：[planner_executor_colocation_5090.md](../wam-inference-acceleration/reports/planner_executor_colocation_5090.md)
