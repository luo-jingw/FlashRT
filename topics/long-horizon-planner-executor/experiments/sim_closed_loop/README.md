# LIBERO 仿真：执行器是否跟随指令

问题：LIBERO-10 上的 ImageWAM 执行器会不会按指令行动，也就是规划器给的子目标能不能改变机器人做什么。

## 结论

**Finding 1：这个执行器几乎不读指令，规划器给的子目标改变不了它做什么。**
只跑执行器、把指令换成 "stay still and do not move" 或换成另一个任务的指令，21 次里仍成功 19 次（原指令 20 次）。
两种替换都只在任务 0 上失败：LIBERO-10 里只有任务 0 和任务 1 共用同一个场景（LIVING_ROOM_SCENE2），
指令只在这里起区分作用；其余 8 个场景里，执行器按场景完成训练时的那一个任务。
因此在这套执行器上，分层只能带来时间开销和偶尔的误导，体现不出规划的收益。

**成功率的差别不可当作结论。** 只跑执行器的 `flat` 与 e8（同样的 episode、噪声种子、步数上限）只差在指令改为 CPU 编码，
21 个回合里就有 3 个翻转了成败（e8 败的任务 2 第 2 回合、任务 8 第 0 回合这里成功，e8 成功的任务 3 第 0 回合这里失败）：
单个回合的成败对微小数值差异很敏感，1–2 个回合的差别在噪声之内。

## 结果

LIBERO-10 任务 0–9 的第 0、1 回合加任务 2 第 2 回合（21 个回合），每个回合 700 控制步上限，执行器单独运行。

| 模式 | 执行器指令 | 成功 | 失败的回合 | 成功回合平均步数 |
|---|---|---|---|---|
| `flat` | 任务原指令 | 20 / 21 | 3:0 | 251 |
| `flat_still` | "stay still and do not move" | 19 / 21 | 0:0、0:1 | 247 |
| `flat_other` | 下一个任务的指令 | 19 / 21 | 0:0、0:1 | 258 |

成败按原任务判定。

## 视频

每个回合一段（640×480，20 fps）：左为 agentview，右为腕部相机。

| 文件 | 内容 |
|---|---|
| `videos/t2e0_flat_still.mp4` | 任务 2，执行器单独运行、指令为 "stay still and do not move"：照样开炉放壶，成功 |

## 设置

| 项 | 取值 |
|---|---|
| 仿真 | LIBERO-10（robosuite 1.4.0，MuJoCo 3.3.2，EGL），渲染 256×256，种子 7，开头 30 步空动作 |
| 执行器 | ImageWAM LIBERO fp8（FlashRT），每次查询执行 10 个动作，噪声种子 1000 × 任务 + 回合；预处理与夹爪映射同 `wam-inference-acceleration/experiments/e8_libero_closed_loop`；加载后释放 FP16 源权重（`colocation_qwen27b_5090/exec_slim.py`），6.38 GB |
| 执行器指令 | `A video recorded from a robot's point of view executing the following instruction: {指令}`，Qwen3-4B 在 CPU 上编码（bf16，每条文本编码一次后缓存），`set_prompt(context=...)` |

## 范围与限制

- 执行器是在 LIBERO 四个套件的完整任务指令上训练的 ImageWAM；它不跟随子目标指令（Finding 1），所以这套环境回答不了
  "分层是否提高成功率"。回答这个问题需要跟随原子指令的执行器，以及不分层会失败的任务。
- 每种模式 21 个回合，每回合一次；成功率差 1–2 个回合不可区分（见结论最后一段）。

## 文件

| 文件 | 内容 |
|---|---|
| `closed_loop_libero.py` | 闭环：执行器、决策逻辑、各模式 |
| `planner_client.py` | 规划器调用：committed-stream 路径、EOS / action 闭合即停、monitor 前缀 capsule |
| `episode_video.py` | 带标注的回合视频 |
| `build_monitor_calls.py`、`calls/` | 从运行记录重建的 monitor 调用（含 `next` 决策或失败的 6 个回合，`../monitor_call/` 使用） |
| `run.sh` / `run_all.sh` | 单次运行 / 全部模式 |
| `runs/<mode>.json` / `.log` | 每种模式每个回合的结果与事件 |
| `videos/` | 上表的视频 |
