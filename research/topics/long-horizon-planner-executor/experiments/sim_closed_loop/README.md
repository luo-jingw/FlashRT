# 规划器 + 执行器在 LIBERO 仿真里的闭环

问题：把 Qwen3.8-27B 规划器（VoLoAgent Full-loop 的决策逻辑）和 ImageWAM 执行器放进同一个闭环，在仿真里跑完整的
"分解 → 执行 → 周期检查 → 切换子目标 / 重规划"，看规划器的决策是否落到机器人动作上、花多少时间、成败有什么变化。

## 结论

**Finding 1：这个执行器几乎不读指令，规划器给的子目标改变不了它做什么。**
只跑执行器、把指令换成 "stay still and do not move" 或换成另一个任务的指令，21 次里仍成功 19 次（原指令 20 次）。
两种替换都只在任务 0 上失败：LIBERO-10 里只有任务 0 和任务 1 共用同一个场景（LIVING_ROOM_SCENE2），
指令只在这里起区分作用；其余 8 个场景里，执行器按场景完成训练时的那一个任务。
因此在这套执行器上，分层只能带来时间开销和偶尔的误导，体现不出规划的收益。

**Finding 2：规划器的两次致命错误都是"看错"，不是"想错"。**
- 任务 0 第 1 回合：分解时把场景里的干扰物认成目标（子目标写成 "small orange-and-red carton … red milk carton"）；
  在任务 0 / 1 共用的场景里，改写后的指令把执行器带去做了任务 1 的动作（抓起黄油盒），第 240 步 monitor 又误判子目标已完成，
  之后 6 次都是 `continue`，700 步超时。只跑执行器时这一回合成功。
- 任务 8 第 0 回合：第 160 步 `next` 之后，执行器抓着第二个摩卡壶 500 步不放，monitor 6 次都判为
  "still achievable / continue"，没有一次 `replan`。
- 成功的回合里也有同类误判（任务 0 第 0 回合）：第 160 步汤罐已放进篮子，monitor 说它"还在桌上"；第 240 步机器人正把番茄酱放进篮子，
  monitor 仍说汤罐在桌上、`continue`；执行器没有等 `next`，自己在第 255 步完成了整个任务。

**Finding 3：同步协议下机器人约 20 % 的时间在等规划器；复用固定前缀 + action 闭合即停降到 12 %，决策与轨迹不变。**
monitor 每 80 个控制步（20 Hz 下 4 s）一次，中位 902 ms，机器人停着等；`planner_fast` 中位 335 ms，
21 个回合的成败、步数、61 次 monitor 决策与 `planner_eos` 逐一相同（`../monitor_call/README.md` 的两项改动在闭环里的效果）。

**成功率的差别不可当作结论。** 只跑执行器的 `flat` 与 e8（同样的 episode、噪声种子、步数上限）只差在指令改为 CPU 编码，
21 个回合里就有 3 个翻转了成败（e8 败的任务 2 第 2 回合、任务 8 第 0 回合这里成功，e8 成功的任务 3 第 0 回合这里失败）：
单个回合的成败对微小数值差异很敏感，1–2 个回合的差别在噪声之内。

## 结果

LIBERO-10 任务 0–9 的第 0、1 回合加任务 2 第 2 回合（21 个回合），每个回合 700 控制步上限。

| 模式 | 执行器指令 | 成功 | 失败的回合 | 成功回合平均步数 | monitor 次数 | monitor 中位 / 最大 (ms) | 规划器耗时 / 回合 (s) | 机器人等待占比 |
|---|---|---|---|---|---|---|---|---|
| `flat` | 任务原指令 | 20 / 21 | 3:0 | 251 | – | – | – | 0 % |
| `flat_still` | "stay still and do not move" | 19 / 21 | 0:0、0:1 | 247 | – | – | – | 0 % |
| `flat_other` | 下一个任务的指令 | 19 / 21 | 0:0、0:1 | 258 | – | – | – | 0 % |
| `planner_eos` | 规划器的子目标 | 19 / 21 | 0:1、8:0 | 233 | 61 | 902 / 2 631 | 3.41 | 19.8 % |
| `planner_fast` | 规划器的子目标 | 19 / 21 | 0:1、8:0 | 233 | 61 | 335 / 898 | 1.88 | 12.0 % |

- 等待占比 = 规划器调用的墙钟时间 /（规划器时间 + 控制步数 / 20 Hz）。指令切换不计入：执行器的 `set_prompt` 中位 1.4 ms；
  本实验把 Qwen3-4B 文本编码器放在 CPU 上（每条新指令约 5 s），这是为了在 32 GB 显存里放下两个模型的做法，不计入也不在视频里停帧。
  文本编码器常驻 GPU 时换一条新指令约 240 ms（`../../../wam-inference-acceleration/experiments/harness_replay_5090/README.md`）。
- 分解：21 个回合里 10 个分成 2 个子目标，1 个分成 3 个，10 个保持 1 个；decompose 中位 520 ms。
- monitor 决策：55 次 `continue`、6 次 `next`，没有 `replan`。
- monitor 阶段中位：`planner_eos` ViT 81 / prefill 314 / 解码 494 ms（52 个 token）；`planner_fast` ViT 41 / prefill 162 / 解码 103 ms（10 个 token，恢复 2 048 个前缀 token）。
  最大值 2 631 ms 是进程里第一次 monitor（录图）。

## 视频

每个回合一段（640×480，20 fps）：左为 agentview，右为腕部相机；下方是当前子目标与最近一次规划器事件。
规划器调用期间机器人不动，视频按调用的墙钟时间停帧并加橙色边框。

| 文件 | 内容 |
|---|---|
| `videos/t6e0_planner_fast_next.mp4` | 任务 6：分成两个子目标，第 160 步 `next` 切到第二个子目标，第 213 步成功 |
| `videos/t6e0_planner_eos_next.mp4` | 同一回合，monitor 解码整段回复：停帧更长，轨迹相同 |
| `videos/t3e0_planner_fast_three_subgoals.mp4` | 任务 3：分成"开抽屉 / 放碗 / 关抽屉"，第 216 步成功 |
| `videos/t0e1_planner_eos_wrong_object.mp4` | 任务 0 第 1 回合：子目标认错物体，执行器去抓黄油盒，超时 |
| `videos/t8e0_planner_eos_stall.mp4` | 任务 8 第 0 回合：第二个摩卡壶抓着不放 500 步，monitor 一直 `continue` |
| `videos/t2e0_flat_still.mp4` | 任务 2，执行器单独运行、指令为 "stay still and do not move"：照样开炉放壶，成功 |

## 设置

| 项 | 取值 |
|---|---|
| 仿真 | LIBERO-10（robosuite 1.4.0，MuJoCo 3.3.2，EGL），渲染 256×256，种子 7，开头 30 步空动作 |
| 执行器 | ImageWAM LIBERO fp8（FlashRT），每次查询执行 10 个动作，噪声种子 1000 × 任务 + 回合；预处理与夹爪映射同 `wam-inference-acceleration/experiments/e8_libero_closed_loop`；加载后释放 FP16 源权重（`colocation_qwen27b_5090/exec_slim.py`），6.38 GB |
| 执行器指令 | `A video recorded from a robot's point of view executing the following instruction: {子目标}`，Qwen3-4B 在 CPU 上编码（bf16，每条文本编码一次后缓存），`set_prompt(context=...)` |
| 规划器 | Qwen3.8-27B master，FlashRT NVFP4，FP8 KV，官方 MTP 头，贪心，关闭思考；committed-stream 路径，解码流读到 EOS 停止 |
| 决策逻辑 | `harness_replay_5090/record_calls.py`（VoLoAgent Full loop）：第 0 步 decompose；每 80 控制步 monitor；`next` 切到下一个子目标，最后一个子目标上的 `next` 或 `replan` 触发 replan（每回合最多 3 次）；工具动作不执行、按 `continue` 处理 |
| 规划器输入 | 仿真的 agentview（作 front 与 exterior）与腕部帧，转正后按 harness 的尺寸加黑边（864×480、1280×720、224×224），JPEG q90 |
| 协议 | 同步：规划器调用期间仿真不前进 |
| 显存 | 执行器 + 规划器共 28.9 GB（已分配） |

`planner_fast`：每回合第 0 步把 monitor 提示词的固定前缀（系统提示 + BEFORE 图，截在 2 048 token）存成 capsule（中位 248 ms，计入规划器耗时），
之后每次 monitor 恢复它、只编码 NOW 图，解码到 `action` 闭合即停；decompose 与 replan 不变。

## 范围与限制

- 执行器是在 LIBERO 四个套件的完整任务指令上训练的 ImageWAM；它不跟随子目标指令（Finding 1），所以本实验只能展示闭环的运转、
  规划器的耗时与错误，不能回答"分层是否提高成功率"。回答这个问题需要跟随原子指令的执行器，以及不分层会失败的任务。
- 渲染分辨率 256×256 的画面放大到 harness 的尺寸后送给规划器，比 harness 的全分辨率画面模糊；规划器的误判可能部分来自这一点。
- 每种模式 21 个回合，每回合一次；成功率差 1–2 个回合不可区分（见结论最后一段）。同一回合在不同进程里跑也可能不同：
  任务 0 第 0 回合在单独的冒烟运行里是 263 步，在批量运行里是 255 步（两次都成功）。

## 文件

| 文件 | 内容 |
|---|---|
| `closed_loop_libero.py` | 闭环：执行器、决策逻辑、五种模式 |
| `planner_client.py` | 规划器调用：committed-stream 路径、EOS / action 闭合即停、monitor 前缀 capsule |
| `episode_video.py` | 带标注的回合视频 |
| `run.sh` / `run_all.sh` | 单次运行 / 全部五种模式 |
| `runs/<mode>.json` / `.log` | 每种模式每个回合的结果与事件（每次规划器调用的阶段时间、回复、决策；每次指令切换） |
| `videos/` | 上表的视频 |
