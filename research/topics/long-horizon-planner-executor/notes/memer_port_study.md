# MemER 代码研读：harness 逻辑与移植到 RoboTwin 2.0

源码：[memer-policy/memer](https://github.com/memer-policy/memer)，commit `fb6dbd0`。论文：arXiv 2510.20328（ICLR 2026）。
下文 `file:line` 均相对仓库根目录。标注 **【判断】** 的是推断或估算，没有源码或论文直接支持。

## 0. 结论先行

| 问题 | 结论 |
|---|---|
| 是否发布了异步 runtime | **没有。** `memer_eval/` 里没有 server、thread、websocket、queue。`MemERDeploymentPolicy.step()` 每调用一次就同步做一次 VLM 推理（`deploy.py:104-142`）。README 的联合部署示例是单线程同步循环（`README.md:289-305`）。异步语义只出现在论文 §3.3。 |
| 高层模型 | 代码和 checkpoint：**Qwen3-VL-4B-Instruct** SFT，1500 steps（`README.md:110`、`TRAINING_QWEN3VL.md:46-71`）。论文：**Qwen2.5-VL-7B-Instruct** SFT 4500 steps，再与原权重做 merge，α=0.8（论文 §3.3 Eq.2、Table 4）。两者不一致。 |
| keyframe 来自哪里 | 部署时来自**模型自己的输出** `keyframe_positions`，经 1D 聚类后写回记忆（`deploy.py:144-151`、`memory.py:57-89`）。训练时用人工规则从 GT 子任务边界生成，属于 teacher forcing（`generate_sft_data.py:216-242`、`:766-783`）。 |
| 低层接入点 | 只有一处：把 `language_command` 从整体任务换成 `current_subtask`（`README.md:250-260`、`:297-300`）。仓库不包含低层代码，只推荐 openpi。 |
| 零样本可行性 | 论文中零样本 GPT-5 和 Gemini-ER-1.5 只做了离线评测（API 延迟 10–15 s），trajectory acc 0.13–0.67，而 MemER 为 0.67–0.87（论文 §4.1 Table 2）。零样本是已知的弱项。 |

## 1. 运行循环（memer_eval/）

### 1.1 代码实际提供的结构

| 组件 | 位置 | 行为 |
|---|---|---|
| 离线评测 CLI | `scripts/eval_subtask_rollout.py:142-147` → `rollout.py:392` `evaluate_rollout` | 单进程。逐 episode、**逐原始帧**循环（`rollout.py:473`），每个 timestep 调一次 `predictor.predict`（`:489`）。不包含低层、环境或时间模型。指标是子任务 exact-match 准确率。 |
| 在线 wrapper | `deploy.py:53` `MemERDeploymentPolicy` | 有状态：持有 `_frames`（所有原始帧，**无上界**，`deploy.py:95,126`）和 `EpisodicMemory`。`step(*obs)` 先 append 全部观测，再**只在最后一帧推理一次**（`deploy.py:104-142`）。 |
| 推理后端 | `inference.py:162-288` `QwenStructuredPredictor` | 进程内 HF `generate`，`do_sample=False`，`max_new_tokens` 默认 128，README 示例用 64（`inference.py:272-277`）。另有抽象基类 `StructuredPredictor.predict(prompt, images)`（`inference.py:29-34`），可以替换成我们自己 runtime 的 client。 |
| 高低层通信 | `README.md:289-305` | 同一循环内直接把 `result.current_subtask` 作为参数传给 `low_level_policy.act(language_command=...)`。 |

### 1.2 论文描述的部署（§3.3、§4）

| 项 | 值 |
|---|---|
| 进程 | "We run both policies on their own server"，高层、低层各一个 server。协议未说明。 |
| 低层频率 | π0.5 约 **2 Hz** 出 chunk。chunk 为 15 个 action @15 Hz，**执行 8 个后 replan**（8/15 s ≈ 0.53 s）。 |
| 高层频率 | 约 **1 Hz**，不是固定周期。图像以 15 Hz 采集，**降采样到 2 Hz** 后入队；上一次预测完成后，立即用最新队列再查询一次。 |
| 高层计算期间 | 低层继续执行，条件是**最近一次预测出的子任务**（"the low-level policy conditions on the latest predicted subtask"）。高层不会打断已下发的 chunk。 |
| 新子任务如何到达 executor | 在低层下一次 replan 时读取最新子任务字符串（≤0.53 s 粒度）。最坏情况下生效延迟约为 L_high + 0.53 s。**【判断】** |
| 时延补偿 | 训练标签取 `t + prediction_horizon·frame_subsample` 帧处的子任务（默认 2×5 = 10 原始帧，`generate_sft_data.py:794-797`、`contract.py:100-110`），让模型**预测略未来的子任务**以抵消推理延迟。若数据为 15 Hz，则提前约 0.67 s。**【判断：数据集 fps 未在代码中写出】** |
| 推理时延 | 论文没给 MemER 本身的数字。32 帧 baseline 约 1 s（§4.1）。 |

**与代码的差异：** 代码中 `frame_subsample=5` 作用在原始帧索引上（`rollout.py:306-313`）。若数据是 15 Hz，上下文就是 3 Hz，8 帧约 2.3 s；论文写的是 2 Hz。**【判断】** 移植时应按真实时间（秒）重新定义 stride。

## 2. 高层输入/输出

### 2.1 Prompt（逐字，`contract.py:11-19`、`:29-41`、`:58-90`）

```
[system]  You are a robot program that predicts actions.
          The video input from the egocentric camera shows the most recent actions the robot has executed. The images are selected frames of particular importance from all the actions the robot has executed so far. Based on these, output the current subtask the robot should execute and nothing else.

          Return a JSON with:
          - current_subtask: the action that should be executed at the current timestep
          - keyframe_positions: list of frame positions (1-indexed) from the video input where actions change
[user]    Task: {instruction}
          Here are the selected frames from the entirety of the full video that are of particular importance:   ← 仅当 memory_count>0
          <image> × M   (M ≤ 8，记忆 keyframe，时间顺序)
          Here is a video of the most recent actions the robot has executed:
          <image> × R   (R ≤ 8，最近帧，时间顺序)
```

- 所有图都以独立的 `{"type":"image"}` 交错插入，**不使用 video 类型**（`contract.py:73-78`）。system 的 content 是 `[{"text":...}]`（`contract.py:87-89`）。
- `instruction` 是一句问句式的任务描述，例如 "What subtask should the robot execute to remove the items ... ?"（`README.md:89`）。
- `system_role` 可设为 `"assistant"`（`inference.py:170`、`contract.py:66`）。

### 2.2 图像布局

| 项 | 值 | 位置 |
|---|---|---|
| 相机 | wrist_left（上）+ exterior_1_left（下），**纵向拼接成一张图** | `configs/robots/panda_stacked_rgb_camera_layout.json`；`rollout.py:234-245` |
| 分辨率 | 每路 320×180，拼接后为 **320×360**（宽 × 高） | `deploy.py:268-272`、`generate_sft_data.py:746-752` |
| 训练像素约束 | `max_pixels 115200`（正好等于 320×360），`min_pixels 50176` | `TRAINING_QWEN3VL.md:60-61` |
| 最近帧 | `recent_frames_length=8`，stride `frame_subsample=5` 原始帧，与当前 t 对齐（`t % 5` 偏移） | `rollout.py:306-313` |
| 记忆帧 | ≤ `memory_length=8`，**排在最近帧之前**，且排除已在最近窗口内的帧 | `memory.py:85-89`、`deploy.py:135-141` |
| 坑 | 单相机时 `_observation_to_pil_image` **不做 resize**，直接用原始分辨率（`deploy.py:264-266`） | |

### 2.3 Keyframe 生成、检索与累积

| 阶段 | 机制 | 位置 |
|---|---|---|
| 训练标签 | 先按子任务标签切出连续段，再用规则文件给每段选 first、last、both 或 none 帧作为 GT keyframe。`keyframe_positions` 是**落在当前最近窗口内的 GT keyframe 的 1-indexed 位置**，所以同一个 keyframe 在窗口内的约 8 步里会被重复标注。 | `generate_sft_data.py:216-242`、`:766-772` |
| 训练时记忆 | 窗口之前的 GT keyframe（对齐到 stride）取最后 8 个，**不经过聚类**。 | `generate_sft_data.py:774-783` |
| 部署：输出 → 绝对索引 | 位置 p 映射为 `context_indices[p-1]`，越界的记入 invalid。 | `rollout.py:328-342`、`deploy.py:146-151` |
| 部署：累积 | 所有候选（保留重复，即"投票"）存入 `_candidate_indices`。 | `memory.py:64-71` |
| 部署：检索 | 先排序，再做 1D single-linkage 聚类：相邻差 ≤ `merge_distance×frame_subsample` = 25 原始帧时合并。每簇取 `median_low` 作为代表，最后保留 8 簇。**票数不参与筛选**，1 票就能成簇。 | `memory.py:21-48`、`:77-83`、`rollout.py:316-325` |
| 是否自产 | **是。** 部署时记忆完全由模型自己的 `keyframe_positions` 闭环产生；解析失败时该步不更新（`deploy.py:146`）。 | |

### 2.4 输出解析（`inference.py:68-141`）

依次尝试三种候选：整段文本、```` ``` ```` 围栏内的内容、第一个配平的 `{...}`。字段 `current_subtask`（别名 `current_primitive`，`contract.py:9`）必须存在；`keyframe_positions` 必须是 list，元素可以是 int、整数值 float 或数字字符串。任一 position 非法时整步失败，返回 `parse_ok=False`，`current_subtask=None`。**部署代码没有失败回退**：README 示例会把 `None` 直接传给低层。移植时必须改成保留上一个子任务。

### 2.5 每次调用的 token 估算 **【判断】**

前提：Qwen3-VL 系视觉编码器 patch=16、spatial merge=2，即每 token 覆盖 32×32 px；`qwen_vl_utils.smart_resize` 按 32 取整；推理时 `do_resize=False`（`inference.py:255`），尺寸由 `process_vision_info` 决定。

| 项 | 计算 | tokens |
|---|---|---|
| 单张 320×360 | 宽 320/32 = 10；高 round(360/32)=11 → 352；10×11 = 110，再加 vision_start/end 2 个 | ≈112 |
| 16 张（8 记忆 + 8 最近，上限） | 16×112 | ≈1,790 |
| system + user 文本 + chat 模板 | 约 90 + 60 + 20 | ≈170 |
| **prefill 合计** | | **≈1,950（上限）**；episode 初期没有记忆时约 1,060 |
| decode | 如 `{"current_subtask":"dust the top shelf","keyframe_positions":[3,4]}` | ≈20–35 |
| RoboTwin 3 相机拼接（3×320×240 → 320×720） | 10×22.5，按 32 取整为 10×22 = 220，加 2 | 每张约 222；16 张约 3,550 + 文本 ≈ **3,700** |

Qwen3.8-27B 若沿用 Qwen3.5/3.6 视觉塔（patch16、merge2），以上数字不变。需要用实际 processor 跑一次 `image_grid_thw` 核对。

## 3. 低层接口

| 项 | 代码 | 论文 |
|---|---|---|
| 输入 | 原始观测 + `language_command=current_subtask`（`README.md:297-300`） | 子任务条件化的 π0.5 |
| 训练 | 用"子任务标签替换 episode 级 instruction"微调（`README.md:252-254`） | 从 **π0.5-DROID** checkpoint 微调 18k steps，batch 128，48 H200·h，另加 10–15 条 intervention 示范（Table 5、§3.3） |
| chunk | 无 | 15 action @15 Hz，执行 8 个后 replan |
| server | 无代码，只推荐 openpi（`README.md:252`） | "own server"，没说明是 openpi websocket。**【判断】** 很可能是 openpi `serve_policy` 的 websocket 服务，但没有证据。 |

## 4. 任务特定 vs 可复用

| 组件 | 类别 | 说明 |
|---|---|---|
| `configs/tasks_from_paper/*_keyframe_rules.json` | 任务特定 | 每个子任务的 regex 或 exact 匹配 → first/last/none。只用于**生成 SFT 标签**，部署时不读取。 |
| 子任务词表与标注（LeRobot `meta/subtasks.parquet`） | 任务特定 | 采集前预先写好子任务列表，操作员按键推进（论文 App. B）。 |
| 微调后的高层权重 | 任务特定 | 子任务字符串、keyframe 判别和输出格式都是**训练出来的**；评测是 exact-match（`rollout.py:500-502`）。 |
| 子任务条件化的低层 π0.5 | 任务特定 | 必须见过同一套子任务措辞。 |
| `instruction` 问句 | 任务特定 | 一句话描述，可直接改写。 |
| `contract.py`（prompt 和 JSON schema） | 可复用 | 零样本时需要扩写（见 §5）。 |
| `memory.py`（候选聚类与可见性） | 可复用 | 与任务无关，只需重设 merge distance 的时间单位。 |
| `inference.py` 的解析器和 `StructuredPredictor` ABC | 可复用 | 换成我们 runtime 的 client 即可。 |
| `deploy.py` 的帧缓冲与上下文索引 | 可复用，需改 | 加帧上限、时间戳，把 stride 改为按秒计。 |
| `camera_layout.py` | 可复用 | 换一份 RoboTwin 的 layout json。 |

## 5. RoboTwin 2.0 移植计划

| # | 组件 | 写或换 | 内容 | 工作量 | 主要风险 |
|---|---|---|---|---|---|
| 1 | Env wrapper | 新写 | RoboTwin 2.0 task → `obs{head, left_wrist, right_wrist}`、`step(action)`、成功判定。按**仿真时钟**推进，不用墙钟。 | M | RoboTwin 控制频率、动作空间与 π0.5 RoboTwin checkpoint 的对齐 |
| 2 | Camera layout | 换 json | 方案 a：head + 1 腕相机纵向拼接成 320×480。方案 b：3 相机拼成 320×720（约 222 tok/图）。同时决定高层与低层是否共用视角。 | S | 双臂任务只给一个腕相机会丢信息；3 相机时 prefill 翻倍 |
| 3 | 高层 predictor | 新写 `StructuredPredictor` 子类 | 调用我们 runtime 服务的 Qwen3.8-27B，记录 prefill/decode 时间。 | S–M | 27B 的 L 远大于 4B，可能达到 1–3 s 量级 **【判断】**，子任务切换滞后 |
| 4 | **零样本 prompt（替代微调）** | 改 `contract.py` | 微调模型隐式学到的东西都要写进 prompt：① **子任务词表**（封闭集合，每个任务一份，供 exact-match 和低层使用）；② 每个子任务的完成判据；③ keyframe 定义（替代 rule json，例如"物体被放下或被看到隐藏信息的那一帧"）；④ 最近帧和记忆帧的时间标注（帧号或秒），因为模型没被训练过理解"1-indexed 视频位置"；⑤ 严格 JSON 输出，可配 guided decoding；⑥ "预测约 0.5 s 后应执行的子任务"（替代 `prediction_horizon` 标签平移）。 | M | 论文中零样本 GPT-5、Gemini 只有 0.13–0.67；27B 开源模型的 keyframe 输出可能噪声很大，记忆闭环会污染自身 |
| 5 | 子任务分解来源 | 新写 | 方案 a：从 RoboTwin 专家脚本 `play_once()` 的阶段调用（grasp/place/move 序列）自动导出子任务词表和每帧标签，同时作为 GT，用于离线复现 `rollout.py` 的准确率评测。方案 b：让 planner 在 t=0 自行分解，开放集合，与低层的对齐风险高。推荐 a。 | M | 专家脚本的阶段粒度可能不等于语言子任务；不同任务的脚本风格不统一 |
| 6 | Keyframe 规则替换 | 新写 | 零样本：由 prompt 定义 keyframe，沿用 `memory.py` 聚类。离线 GT：按 #5 的阶段边界加每任务的 first/last 规则（格式复用 `generate_sft_data.py:140-213`）。对照：用规则直接产 keyframe（oracle memory），隔离 planner 的 keyframe 误差。 | S | RoboTwin 多数任务记忆依赖弱，需要挑或构造真正需要记忆的任务（计数、搜索、重复放置） |
| 7 | 跟随子任务的 executor | 微调或新写 | π0.5、ImageWAM 的 RoboTwin checkpoint 通常是**整任务 instruction** 条件化的，不会遵循子任务指令。需要用 #5 的子任务标签替换 instruction 重新微调（MemER 的做法），或者先测零样本对子任务措辞的服从度。chunk 与 replan 参照 15/8。 | **L** | 最大风险：没有子任务微调，低层忽略高层输出，harness 就没有效果；ImageWAM 是否支持语言子任务条件也需要确认 |
| 8 | 异步调度与延迟注入 | 新写（代码中没有） | 在仿真时钟上用离散事件模拟：高层在 t₀ 取快照（最近帧、记忆），计算时间 L（实测或注入的值）；结果在 t₀+L 写入"latest subtask"寄存器，低层在下一个 replan 边界读取。L 期间到达的帧用 `step(*chunk)` 批量入缓冲（`deploy.py:104-127` 已支持）。上一次完成后立即开始下一次查询。 | M | 墙钟与仿真时钟混用会导致不可复现；GPU 共享时 L 会被低层推理干扰 |
| 9 | 日志 | 新写 | 每次高层调用记录：`t_obs_snapshot`、`t_request`、`t_prefill_end`、`t_decode_end`、`L`、`t_effective`（低层首次使用的 sim 时刻）、prefill/decode token 数、`image_grid_thw`、raw_text、parse_ok、子任务、keyframe 映射、记忆索引（字段沿用 `rollout.py:518-537` 的 row）。低层记录每个 chunk 的 `t`、使用的子任务和推理时延。 | S | 无 |
| 10 | 鲁棒性修补 | 改 `deploy.py` | 解析失败时保留上一个子任务；给 `_frames` 加上界（只保留记忆候选和最近窗口）；单相机时也 resize。 | S | 无 |

**MemER 假定高层已微调的位置（零样本必须替代）：**
1. 子任务措辞来自训练集，是封闭词表，并且与低层共享（`rollout.py:500-502` 用 exact-match）。零样本需要在 prompt 里给出词表。
2. `keyframe_positions` 的语义来自规则标签（`generate_sft_data.py:766-772`），system prompt 只写了 "where actions change"。零样本需要具体定义 keyframe。
3. 未来子任务偏移（`prediction_horizon`）是通过标签平移学到的。零样本需要在 prompt 里显式说明。
4. 输出格式的稳定性：微调模型几乎不出格式错误（参考 94.21% acc，`README.md:241-246`）。零样本需要 guided JSON 加失败回退。
5. 训练时记忆是 GT keyframe，没有自产噪声；部署时是自产的。微调模型对这个分布偏移已有一定鲁棒性，零样本模型没有。

## 6. 仓库事实

| 项 | 值 |
|---|---|
| License | **仓库中没有 LICENSE 文件**，README 也没提到。默认保留所有权利，复用代码前需确认授权。 |
| 最后提交 | `fb6dbd0` 2026-03-30 "initial code release"，**全仓库只有 1 个 commit** |
| 大小 | 工作区 180 KB，.git 220 KB；源码共 3,150 行（`memer_eval/` 7 个模块约 1,390 行；`scripts/generate_sft_data.py` 909 行） |
| 依赖 | `pyproject.toml`：python ≥ 3.11，numpy==2.2.6，Pillow，lerobot==0.4.4，tqdm==4.67.3。extra `eval` 加 transformers ≥ 4.57.0 和 qwen-vl-utils。torch 自备。训练使用外部的 QwenLM/Qwen3-VL finetune 代码。 |
| 发布资产 | HF checkpoint `ajaysri/memer-dusting-qwen3vl-4b-step-1500`；数据集 `ajaysri/dusting_{train_50,test_10}_v3_subtasks_rgb`（LeRobot v3） |
| 不包含 | 低层策略代码、server 和异步 runtime、Object Search 与 Counting 的数据和权重、论文中的 model merge 步骤 |
