# long-horizon-planner-executor

长时程机器人任务中的"规划器 + 执行器"双模型系统：为什么长时程任务要用两个模型，两模型放在同一块 Jetson AGX Thor 上时有哪些普通 LLM 服务没有的挑战，用什么目标指标衡量，以及这些挑战可能怎么解决。

- 状态：第一轮综述完成（快照 2026-11-09）；monitor 调用实验与 LIBERO 仿真闭环（RTX 5090）完成，Thor 实测待做（2026-11-10）
- 最后更新：2026-11-10

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
| [survey.md](survey.md) | 动机（长时程定义、同执行器对照、反证、为什么两个模型）、延迟代价的三条途径与指标、边缘 SoC 的带宽与 EMC 现象、系统论文的写法、开放问题、阅读顺序 |
| [papers.md](papers.md) | 论文目录，按 A 动机 / B 延迟代价 / C 系统现象 / D 写法参照分节 |
| `notes/` | 精读笔记：[2026-vla-xpu-characterization.md](notes/2026-vla-xpu-characterization.md) |
| `experiments/` | [monitor_call](experiments/monitor_call/README.md)：monitor 调用的固定前缀（61–65 % 的输入）与决策位置（第 9–11 个 token），复用前缀 + action 闭合即停后到决策可用的时间（RTX 5090），以及不对齐的前缀边界翻转干预决策；[sim_closed_loop](experiments/sim_closed_loop/README.md)：规划器 + ImageWAM 在 LIBERO 仿真里的闭环（21 个回合 × 5 种模式，带标注视频）：执行器几乎不读指令，规划器的失败来自看错，同步协议下机器人约 20 % 的时间在等规划器 |
| [summary_1111.html](summary_1111.html) | 11/05–11/11 进展汇报：按场景动机、Thor 上的 challenge、解决方向三个问题组织（论文证据卡片、challenge 排序、Thor 实测、闭环视频） |
| [findings_1112.html](findings_1112.html) | 六个发现，每条一张由 Thor 原始日志画出的图：Thor 与 RTX 5090 的规划器延迟、瓶颈阶段随 harness 变化、共置时静态划分顾不了两头、干扰随规划器阶段变化、SM 分区共享芯片级吞吐上限、执行器的 GPU 时间占比 |
| `reports/` | 汇报材料（尚无） |

## 相关记录

- Thor 实测：根目录 `THOR_STATUS_SUMMARY.md`（`1104_h3`、`1104_h3b`、`1104_h3c`、`1105_h6h4` 轮）
- 空闲后解码变慢：根目录 `issues.md` → ISSUE-112
- 待核实的论文元数据：根目录 `issues.md` → ISSUE-113
- prefill 分块切分改变回复与决策：根目录 `issues.md` → ISSUE-110
- Thor 上 monitor 调用的复用与闭合即停：根目录 `THOR_CHECKLIST.md` → H8
- RTX 5090 共置报告：[planner_executor_colocation_5090.md](../wam-inference-acceleration/reports/planner_executor_colocation_5090.md)
