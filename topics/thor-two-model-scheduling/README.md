# thor-two-model-scheduling

一块 Jetson AGX Thor 上，周期执行器（约 4B 的 VLA/WAM，200 ms 时限）与 VLM 规划器共置时的推理速度与调度。在 [edge-colocation-scheduling](../edge-colocation-scheduling/README.md) 第一轮综述之上补三个缺口：Apple 自家的系统工作、Jetson Orin / Thor 上的 VLA/VLM 实测、机器人以外的快慢双模型系统。属于主线 [long-horizon-planner-executor](../long-horizon-planner-executor/README.md) 的调研方向二。

- 状态：立项与第一轮论文卡片（9 篇），快照 2026-11-19
- 最后更新：2026-11-19

## 文件

| 文件 | 内容 |
|---|---|
| [brief.md](brief.md) | 立项：为什么做与主线的关系、要回答的问题与判读表、已有工作、设置、代价与停止条件、结果（待填） |
| [papers.md](papers.md) | 论文卡片 9 篇（AFM 2024、AFM 2025、ReDrafter；EagleVLA、Characterizing VLA、Offload or Overload；DriveVLM、AsyncDriver、Real-Time Reasoning Agents）：问题与解决程度、方法来源、硬件与调度画像、与我们的重合度；判读表各问题的答案与未覆盖的空白 |

相关方向：[planner-call-latency](../planner-call-latency/README.md)（规划器单次调用本身的加速）。
