# Thor 上线检查：规划器权重、规划器 monitor 调用、π0.5 时延

问题：在共享的 Jetson AGX Thor（主机 `thortwo`）上，规划器加执行器这条线能否直接开始实验？具体回答三件事：
1. Thor 上的规划器权重是否与 RTX 5090 那份逐张量相同；
2. 规划器在 Thor 上跑 monitor 调用，结果是否与以前一致；
3. π0.5 执行器在 Thor 上单独运行一次要多久。

## 结论

1. **规划器权重已与 RTX 5090 那份逐张量相同。**
   - 做法：把 5090 主机上 `qwen38-27b-nvfp4-official`（由官方 `Qwen/Qwen3.8-27B-FP8` 转换而来）拷到 Thor 的 `~/thor_val/ckpt/qwen38-27b-nvfp4-official-5090`。
   - 核对：在 Thor 上逐张量计算哈希，1 696 个张量，哈希列表的 sha256 为 `7c01884414cd…`，与 5090 和 Chenwei 那两份相同。
   - 之后 Thor 上统一用这一份，有 63 个张量不同的旧副本（ISSUE-115）不再使用。
2. **规划器 monitor 调用与以前一致。**
   - 设置：wy_lt prefill，回合 0 / 43，腕部相机 224，共 20 次调用。
   - 对齐复用加闭合即停（ali/sto）时，从调用开始到决策可用为 1.75–1.85 s，与 `monitor_call/README.md` Finding 5 的 1.77 s 一致。
   - ali/sto、ali/dec 两组各 20 次决策，与该次运行自己的 `generate` 全部相同。
3. **π0.5 单独运行一次的 P50 为 41.0 ms**。设置为两路相机、10 步动作，测 50 次，最小 40.9 ms、最大 41.7 ms。

## 设置

| 项 | 取值 |
|---|---|
| 提交 | `7373e8f3`（`out/00_state.log`） |
| 系统 | L4T R38.2.2；MAXN；未改时钟，未用 sudo |
| GPU 锁 | 每个 GPU 步骤都通过 `colocation_thor/with_gpu_lock.sh` 运行；规划器 01:06–01:30，π0.5 01:30–01:31（2026-11-18，美东时间） |
| Python | `~/workspace/pi0.5/FlashRT/.venv`（torch 2.9.1+cu130）。规划器另把 `~/thor_val/pydeps_tf5`（transformers 5.17）放在 `PYTHONPATH` 最前；venv 自带的 transformers 4.53 没有这个结构的 processor |
| 规划器 | `PLANNER_CKPT=~/thor_val/ckpt/qwen38-27b-nvfp4-official-5090`，`PLANNER_MTP_DIR=~/thor_val/ckpt/qwen38-27b-fp8-official`，`FLASHRT_QWEN36_THOR_LIN_PREFILL=parent`，`FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND=wy_lt` |
| π0.5 | `examples/quickstart.py --benchmark 50 --warmup 300`，权重为 `~/workspace/pi0.5/logs/0924_apxinf_pi05/ckpt`（带 norm stats 的 π0.5 PyTorch 权重；本步只测时延）。HF 上的 `lerobot/pi05_libero` 没有附带 norm stats，FlashRT 无法直接加载 |

## 文件

| 文件 | 内容 |
|---|---|
| `run_bringup.sh` | 依次运行：记录板子状态、逐张量哈希、规划器 monitor 调用、π0.5 时延 |
| `out/00_state.log` | 提交、L4T、`nvidia-smi`、时间 |
| `out/01_hash.log` | 张量数与哈希列表的 sha256 |
| `out/02_monitor_call.json`、`out/02_monitor_call.log` | 规划器 20 次调用，每种模式的时延与决策是否改变（列的含义见 `../monitor_call/README.md`） |
| `out/03_pi05.log` | π0.5 时延 |
