# 立项：Thor 上两模型的推理速度与调度

快照 2026-11-19。范围：一块统一内存 SoC 上，周期执行器（约 4B 的 VLA/WAM，200 ms 时限）与 VLM 规划器（Qwen3.8-27B）共置时的推理速度与调度；包括同类硬件（Apple M 系列等）上的系统工作，以及机器人以外的快慢双模型系统。单个规划器调用本身的加速见 [planner-call-latency/brief.md](../planner-call-latency/brief.md)。

## 1. 为什么做，和主线的关系

主线问题：异步下规划器的决策延迟 L 怎样影响任务质量（[tasks_robocerebra.md](../long-horizon-planner-executor/tasks_robocerebra.md)）。共置时，L 还受另一个模型的干扰。

- 已实测（Jetson AGX Thor，B27 / B28）：执行器单独 p99 37.5 ms；共置（MPS 客户端优先级）p99 70 ms，0 % 错过 200 ms；CPU 满载访存时规划器 decode 慢约 2.2 倍，执行器 p99 约翻倍。
- 执行器余量大（约 60–160 ms），规划器余量为负（MemER 节奏下 L ≈ 2 s > 1 s）。
- 未证实：调度能把干扰带来的损失要回来。零拷贝基本无收益（discrete 流程的拷贝只在加载时发生）。

不做：不知道执行器多出来的余量能否换成规划器的 L，主线的"系统贡献"无从谈起。

## 2. 要回答的问题与判读

| 问题 | 答案 → 影响的决定 |
|---|---|
| 统一内存设备上，有没有工作把内存带宽当作调度对象，且有截止期？ | 有：作为基线和对照；没有：空白成立，记为候选问题 |
| Apple 是否发表过自家硬件上的推理系统或调度工作？ | 有：补进对比，看他们怎么处理 CPU/GPU/ANE 共用带宽；没有：只保留 MLX、EdgeAgent 等间接证据，写明检索范围 |
| Jetson Orin / Thor 上部署 VLA/VLM 的工作测了什么、做了什么？ | 决定我们的测量是否已有人做过、哪些数字可作对照 |
| 快慢双模型系统（自动驾驶、机器人以外）用什么指标衡量慢模型的时延对质量的影响？ | 决定排序指标的写法 |

## 3. 已有工作

本方向大半已有调研，不重做：

- [../../edge-colocation-scheduling/survey.md](../edge-colocation-scheduling/survey.md)：略读 30 篇，精读 5 篇，分 GPU 共享与抢占、LLM 阶段调度、边缘 SoC 带宽争用、机器人双系统部署四类。
- [../../edge-colocation-scheduling/unified_memory_inference.md](../edge-colocation-scheduling/unified_memory_inference.md)：统一内存设备上的推理框架与系统 16 篇，含 MLX、LLM in a flash、EdgeAgent（M4）、Agent.xpu（按优先级让出带宽）。

本轮只补三个缺口（论文卡片见 [papers.md](papers.md)）：

1. Apple 自家的系统工作（基础模型技术报告、投机解码、ANE 调度）；
2. Jetson Orin / Thor 上的 VLA/VLM 部署与实测；
3. 机器人以外的快慢双模型系统（如自动驾驶双系统）。投机解码只作为工具引用，不算两模型共置。

## 4. 设置

每篇论文记录：硬件是否统一内存；有无截止期；调度单位（kernel、请求、阶段）；两模型是否同时在一块芯片上；评估指标。用这些字段与 Thor 场景对照。

Thor 场景的固定事实：20 个 SM；内存带宽 273 GB/s，CPU 与 GPU 共用；规划器 NVFP4 权重 25.8 GiB，64 层里 16 层全注意力；执行器 ImageWAM 8.5 GiB；batch = 1。

## 5. 代价与停止条件

- 每个缺口精读 2–3 篇，宁少勿滥；略读不超过 15 篇。
- 停止：某个缺口检索不到直接相关工作时，写明检索词与范围后结束，不扩大到间接相关的工作。

## 6. 结果

（待填：每个问题的答案；对应判读表的哪一行；结论与置信度；去向。）
