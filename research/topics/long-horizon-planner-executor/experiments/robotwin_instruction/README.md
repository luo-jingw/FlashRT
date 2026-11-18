# ImageWAM 执行器在 RoboTwin 2.0 上是否使用语言指令

问题：ImageWAM-FLUX.2-4B-RoboTwin 执行器在 RoboTwin 2.0 上是否真的读指令，还是像在 LIBERO 上那样只看场景
（`../sim_closed_loop/README.md` Finding 1：LIBERO 上换成 "stay still and do not move" 或别的任务指令仍 19/21 成功）。

## 结论

**RoboTwin 上的执行器依赖指令：去掉指令（"stay still" / 空串）或换成别的任务的指令，6 个任务里 5 个从 9–10/10 掉到 0/10，
`click_bell` 也从 10/10 掉到 3–6/10；与 LIBERO 相反。** 但它读的是"拿哪个物体、做哪类动作"，不读方位词：
`place_a2b_left/right` 把同一物体对的指令里的 left 换成 right（反之亦然）仍 9/10、10/10 成功，放置方向由场景布局决定
（两个任务的场景生成器里 A 总在 B 的目标一侧）；而不给指令时它常去抓 B 而不是 A。
置信度：高（每格 10 回合，原指令与无指令的差距是 9–10 vs 0；同种子成对比较里没有一个回合在替换臂成功而原指令失败）；
"不读方位词"这一条只来自一对任务，置信度中等。

## 结果

每格：成功 / 回合，每臂 10 回合，同一起始种子（4300000 起，经专家可解检查后的种子序列）。

| 任务 | 原指令 | "stay still and do not move" | 另一任务的指令 | 空串 |
|---|---|---|---|---|
| place_a2b_left | 9/10 | 0/10 | 9/10（换成 place_a2b_right 指令，同一物体对） | 0/10 |
| place_a2b_right | 10/10 | 0/10 | 10/10（换成 place_a2b_left 指令，同一物体对） | 0/10 |
| move_can_pot | 9/10 | 0/10 | 0/10（lift_pot 指令，同一个锅） | 0/10 |
| lift_pot | 10/10 | 0/10 | 0/10（move_can_pot 指令，场景里无罐子） | 0/10 |
| click_bell | 10/10 | 6/10 | 5/10（beat_block_hammer 指令） | 3/10 |
| beat_block_hammer | 10/10 | 3/10 | 0/10（click_bell 指令） | 0/10 |

- 成功回合平均步数（原指令 / 替换臂）：place_a2b_left 131 / 132（other），place_a2b_right 135 / 141（other），
  click_bell 53 / 74（still）、59（other）、93（empty），beat_block_hammer 104 / 102（still）。
- 同种子成对：替换臂成功而原指令失败的回合数在所有格子里都是 0。少数格子与原指令只共享 9 个种子：
  RoboTwin 的专家可解检查在不同臂之间偶尔结果不同，种子序列随之错开一位（`results.json` 的 `seeds_common_with_original`）。
- 视频里看到的行为（每臂前 2 回合）：
  - `place_a2b_left` still / empty：机械臂去抓场景里的 B（茶盒）而不是 A（肥皂），超时。
  - `move_can_pot` other（"lift the kitchenpot with both arms"）：双臂一起伸向锅，对应 lift_pot 的动作，没有去抓罐子。
  - `beat_block_hammer` other（"click the bell"）：手臂悬停在锤子上方，不抓取。
- 全部 240 回合 GPU 墙钟约 70 分钟（每任务 6–17 分钟，单进程加载一次模型约 12–17 s）。

## 设置

| 项 | 取值 |
|---|---|
| 仿真 | RoboTwin 2.0（`tmp/ImageWAM/third_party/RoboTwin`），`demo_clean`，aloha-agilex，步数上限取 `_eval_step_limit.yml`（这 6 个任务均为 400） |
| 执行器 | 官方 torch ImageWAM 策略（`tmp/ImageWAM/experiments/robotwin/imagewam_policy`），检查点 `tmp/checkpoints/imagewam_release/robotwin/flux2_klein_4b/model.pt`（HF `yuyangalin/ImageWAM-FLUX.2-4B-RoboTwin`，step 69900），bf16，串行 10 步动作去噪，action_horizon 16 / replan 16，`compact_288x256`，text_cfg_scale 1.0，Qwen3-4B 文本编码器在 GPU 上；设置与 `wam-inference-acceleration/experiments/robotwin_aog_subspace/run_task.sh` 相同（不设 `IMAGEWAM_ACTION_STEP_SUBSET`） |
| 指令 | 执行器收到 `A video recorded from a robot's point of view executing the following instruction: {指令}`；原指令来自评测器的 unseen 模板（`instruction_type: unseen`） |
| 四个臂 | `original`：评测器原指令；`still`："stay still and do not move"；`other`：见下；`empty`：空串 |
| `other` | `task_specs.py`。场景里有对方任务的物体时，用对方任务的 unseen 模板填本回合的物体（place_a2b_left↔right，move_can_pot→lift_pot）；否则用对方任务的一句固定指令（lift_pot→"Pick up the sauce can and set it beside the kitchen pot."，click_bell→"Grab the hammer and beat the block."，beat_block_hammer→"Click the top center of the bell on the table."） |
| 种子 | `seed=42`（起始 4300000），各臂相同；评测器在 `setup_demo` 里按回合种子设 numpy / torch 种子；物体描述选择所用的 Python `random` 在 RoboTwin 里没有设种子，这里按回合种子设定，使各臂的原指令文本一致 |
| 环境 | RTX 5090，`tmp/robotwin_venv`（flashrt conda env 上的 overlay，torch 2.14 cu130），运行前 GPU 空闲（391 MiB，0 %） |
| 代码版本 | FlashRT `f209da67`（工作区有未提交改动，与本实验无关）；ImageWAM `5d4a341` 加 PROJECT.md 记录的本地改动（`deploy_policy.py` 的 action-step-subset 钩子，本实验不启用） |

替换方式：每臂调用 RoboTwin 自己的 `script/eval_policy.py:eval_policy` 循环；评测器照常生成原指令后，
`run_instruction_arms.py` 在任务环境实例上包装 `set_instruction`，把指令换成该臂的指令，因此场景、种子与评测器的随机数消耗在各臂之间一致。
没有改动 FlashRT 库代码和 ImageWAM / RoboTwin 源码。

命令：

```bash
cd /home/jingwu/workspace/FlashRT/research/topics/long-horizon-planner-executor/experiments/robotwin_instruction
ln -sf ~/anaconda3/envs/flashrt/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2 bin/ffmpeg
./run_all.sh 10 42 2 main > logs/main_suite.log 2>&1       # 6 任务 × 4 臂 × 10 回合，每臂前 2 回合录视频
~/anaconda3/envs/flashrt/bin/python collect_results.py runs/main results.json
```

`run_task.sh` 内部：`source tmp/env_5090.sh; source tmp/robotwin_env.sh`，在 RoboTwin 根目录下运行
`$ROBOTWIN_PY run_instruction_arms.py TASK EPISODES SEED OUT_DIR VIDEO_DIR VIDEO_EPISODES ARMS`。

## 范围与限制

- 每格 10 回合，只用 `demo_clean`；`click_bell` 的 3–6/10 之间的差别不可区分。
- `place_a2b_left/right` 的 `other` 成功说明执行器不按方位词放置，但两个任务的场景布局本身就把 A 放在目标一侧，
  场景与原任务一致，无法区分"读到了 left/right 但被场景压过"和"根本不读方位词"。
- 视频帧来自评测器的 `now_obs`，在 `skip_get_obs_within_replan` 下每 16 个控制步才更新一次，视频是逐块跳变的。
- `other` 臂在 place_a2b / move_can_pot 上墙钟较长（约 7 分钟 / 10 回合）：模板生成调用沿用了 RoboTwin 默认的
  `max_descriptions=1000000`，每回合生成大量重复字符串；不影响结果。

## 文件

| 文件 | 内容 |
|---|---|
| `task_specs.py` | 6 个任务与各自 `other` 指令的来源 |
| `instruction_arms.py` | 四个臂与 `InstructionSelector` |
| `official_usr_args.py` | 按 `eval_robotwin_single.py` 的方式组出 RoboTwin 评测参数 |
| `robotwin_task_args.py` | 按 `eval_policy.main` 的方式组出任务 `args`，每臂独立输出目录与视频目录 |
| `run_instruction_arms.py` | 单任务、多臂、一次加载模型的驱动 |
| `run_task.sh` / `run_all.sh` | 单任务 / 全部任务 |
| `collect_results.py` | 汇总为 `results.json` 并打印表格 |
| `results.json` | 表格（含成对计数、成功回合平均步数）与每回合记录（任务、臂、种子、成功、步数、给出的与原指令、视频路径） |
| `runs/main/<task>/<arm>/` | 评测器的 `episode_end_states.jsonl` 与 `arm_episodes.jsonl` |
| `logs/` | 每任务日志与 `main_suite.log` |
| `videos/main/<task>/<arm>/` | 视频（头部相机 + 双腕相机，10 fps），文件名带成败。仓库只收录 place_a2b_left、move_can_pot、beat_block_hammer 三个任务的 original 与一个替换臂（still 或 other）各第 0 回合；每臂前 2 回合的全部 48 段只留在运行机器上 |
| `bin/ffmpeg` | 指向 imageio_ffmpeg 自带 ffmpeg 的符号链接（本机无系统 ffmpeg） |
