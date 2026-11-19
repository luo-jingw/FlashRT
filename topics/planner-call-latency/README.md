# planner-call-latency

多帧上下文（1–3k token）下，VLM 规划器单次调用怎么变快：时间花在图像编码、prefill、解码哪一段，各段有哪些已有办法，哪些在 Qwen3.8-27B（48 层线性注意力 + 16 层全注意力）与 Jetson AGX Thor 上不成立。属于主线 [long-horizon-planner-executor](../long-horizon-planner-executor/README.md) 的调研方向一。

- 状态：立项与第一轮论文卡片（7 篇），快照 2026-11-19；Thor 上的分段实测（J5）待做
- 最后更新：2026-11-19

## 文件

| 文件 | 内容 |
|---|---|
| [brief.md](brief.md) | 立项：为什么做与主线的关系、要回答的问题与判读表、已有工作、设置、代价与停止条件、结果（待填） |
| [papers.md](papers.md) | 论文卡片 7 篇（VisionZip、VLA-Cache、StreamingVLM、CacheBlend、VLCache、SGLang、LinearKV）：intro 的问题与解决程度、方法是否来自非显然观察、与我们的重合度及他们没考虑到的地方；综合与未覆盖的空白 |

相关方向：[thor-two-model-scheduling](../thor-two-model-scheduling/README.md)（两模型共置时的调度）。
