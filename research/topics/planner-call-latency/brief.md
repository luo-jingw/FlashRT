# 立项：多帧上下文下，规划器单次调用怎么变快

快照 2026-11-19。范围：单个 VLM 规划器在 1–3k token 的多帧上下文下，一次调用从取观测到给出决策的时间；不含两模型共置的调度（见 [thor-two-model-scheduling/brief.md](../thor-two-model-scheduling/brief.md)）。

## 1. 为什么做，和主线的关系

主线问题：异步下规划器的决策延迟 L 怎样影响任务质量（SR、完成时间）（[tasks_robocerebra.md](../long-horizon-planner-executor/tasks_robocerebra.md)）。L 的主体是规划器单次调用的时间。

- MemER 的高层约 1 Hz 调用一次，每次看近期 8 帧与至多 8 张关键帧，约 1.2–1.8k token。
- Jetson AGX Thor 实测（2026-11-19，HF 官方 Qwen3.8-27B，关思考）：2k token 时 prefill 1.12 s；回复 10–18 token 的整次调用 1.0–1.6 s，回复约 100–150 token 的 4–6 s。
- 文献给出的上下文需求是几百到 2k token（MemER 关键帧记忆 ≤ 约 1.8k token，SR 96.3 %），所以问题不是"长上下文"，而是"够用长度下单次调用仍慢于调用节奏"。

不做：L 的下限停在 1–4 s，主线实验（C4、J4）只能在比 MemER 慢得多的节奏下进行。

## 2. 要回答的问题与判读

问题：1–2k 多帧上下文下，一次调用的时间花在哪一段，每一段文献里有哪些成熟的削减办法，哪些办法在我们的模型和场景下不成立。

时间拆成：图像编码（ViT）、prefill、解码（输出 token 数 × 每 token 时间）、固定开销（graph 录制、调用启动）。拆分来自 J5 的 Thor 实测。

| 结果 | 说明什么 | 下一步 |
|---|---|---|
| 图像编码 + prefill 占一半以上 | 相邻两次调用的帧大部分相同，重复计算是主要浪费 | 精读跨调用复用（流式视频 KV 复用、非前缀 KV 复用）与视觉 token 压缩 |
| 解码占一半以上 | 输出长度决定 L | 精读输出侧：结构化短输出、决策字段出来即停、投机解码在 MTP 之上的增量 |
| 固定开销占显著比例 | 不是算力问题 | 转入 FlashRT 工程问题（graph 按位置录制，ISSUE-117 一类），不再找文献 |
| 文献的办法都依赖全注意力 KV 的可拼接性 | Qwen3.8-27B 64 层里 48 层是线性注意力（GDN），状态不能按位置拼接 | 这本身是一个空白，记为候选问题 |

## 3. 已有工作

- 本组：monitor 调用实验（[../experiments/monitor_call/README.md](../long-horizon-planner-executor/experiments/monitor_call/README.md)）中，"前缀复用 + 决策闭合即停"让 Thor 上一次 monitor 调用从 6.40 s 降到 1.77 s；MemER 规划器在 RTX 5090 上 8 帧输入 1 207 token，暖调用约 0.3 s（[../harness/README.md](../long-horizon-planner-executor/harness/README.md)）。
- 文献类别（论文卡片见 [papers.md](papers.md)）：视觉 token 剪枝与压缩；流式视频 LLM 的 KV 复用；非前缀位置的 KV 复用；输出侧加速（投机解码、早停）。

## 4. 设置

- 模型：HF 官方 Qwen3.8-27B，FlashRT NVFP4，FP8 KV，MTP 投机解码；关思考。
- 机器：Jetson AGX Thor（共享，经 GPU 锁），graph 缓存上限 `FLASHRT_QWEN36_GRAPH_CACHE_MAX=8192`。
- 提示词：MemER 零样本提示词（[../harness/memer_prompt.py](../long-horizon-planner-executor/harness/memer_prompt.py)），近期 8 帧（每 0.5 s 一帧）+ 至多 8 张关键帧，每帧两路相机 320×180 竖拼。
- 调用节奏参照：MemER 高层约 1 Hz（论文值）。
- 论文筛选：只收与"单个 VLM、batch = 1、多帧或流式输入、低时延"直接相关的工作；纯吞吐（大 batch 服务）工作不收。

## 5. 代价与停止条件

- J5 拆分实测约 1 小时 Thor 机时。文献精读 6–8 篇，宁少勿滥。
- 停止：拆分显示单次调用已在 1 s 左右，本方向降级为记录；某类办法在线性注意力层上不成立且无替代时，只记空白，不展开。

## 6. 结果

（待填：J5 拆分实测值；每类办法对应判读表的哪一行；结论与置信度；去向。）
