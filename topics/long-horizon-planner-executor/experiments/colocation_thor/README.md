# 规划器进程：按 harness 节奏回放 27B 规划器，供 Thor 上的共置测量

问题：Thor 上执行器与真实 27B 规划器共置时，执行器的截止期与规划器的"从调用开始到决策可用"各是多少。本目录提供规划器一侧：
一个独立进程按 VoLoAgent harness 的节奏回放录制的调用，每次调用记下开始、决策可用、结束三个时间戳（单调时钟与墙钟），
供与执行器的日志对齐。执行器、共享方式与 Thor 上的 GPU 独占锁由共置测量一方提供。

## 用法

```
PLANNER_CKPT=DIR PLANNER_MTP_DIR=DIR python planner_process.py {rtx|thor} {original|fast} OUT_JSONL CALLS_JSONL...
```

| 项 | 取值 |
|---|---|
| 调用节奏 | 每回合开始时 decompose；第 s 步的 monitor 在回合开始 + s / 80 × 5.33 s 到期；上一次调用未结束时顺延（单一规划器流） |
| `original` | 每次调用整段 prefill、解码到 EOS，决策在回复结束时可用 |
| `fast` | decompose 之后建一次对齐前缀的 capsule（2 048 token）；monitor 恢复 capsule、只 prefill 其余 token、`action` 闭合即停 |
| 热身 | 计时之前先不按节奏、不输出地把所有回合跑一遍：CUDA graph 按提示词长度与解码位置惰性录制，热身之后计时的一遍不含录制 |
| 决策可用 | 两组都取输出闭合 `action` 值的那个解码周期结束时；没有 `action` 值的回复（decompose 的编号列表）取回复结束。L_decision = 决策可用 − 开始，L_reply = 结束 − 开始 |
| 输出 | 每次调用一行 JSON：到期 / 开始 / 决策可用 / 结束时刻，各阶段 ms，新 token 数，决策；capsule 构建单独一行 |
| GPU 锁 | Thor 上经 `with_gpu_lock.sh` 运行：只读打开 `/tmp/lingbot_va_gpu_exclusive.lock`，`flock(LOCK_EX)` 持有到进程结束（与共置测量一方的约定） |

## 结果（RTX 5090，官方 Qwen3.8-27B，6 段录制、66 次调用）

功能检查：ep0 / ep43 / ep188 × 腕部 224 / 720，6 次 decompose、59 次 monitor、1 次 replan。GPU 上另一个用户的进程整段在跑
（约 5.7 GB，利用率 37–45 %），耗时是**共享 GPU** 下的数。中位（最大），单位 s。

| 组 | 调用 | L_decision | L_reply | 开始延迟 | 新 token |
|---|---|---|---|---|---|
| `fast` | monitor 224（30） | 0.457（0.530） | 同左 | 0.000 | 14 |
| `fast` | monitor 720（29） | 0.664（0.737） | 同左 | 0.000 | 14 |
| `original` | monitor 224（30） | 0.720（0.925） | 1.443（2.072） | 0.000 | 70.5 |
| `original` | monitor 720（29） | 0.939（1.164） | 1.638（2.318） | 0.000 | 67 |
| 两组 | decompose（6） | = L_reply | 0.63–1.17（`fast` 一组） | 0.01–0.02 | 38–75 |
| 两组 | replan（1） | = L_reply | 0.29（`fast` 一组） | 0.64–1.36 | 14 |

- 两组 59 次 monitor 决策逐次相同：`continue` 53、`replan` 5、`next` 1。decompose 与 replan 的回复没有 `action` 值，决策可用即回复结束。
- replan 与同一步的 monitor 同时到期，排在 monitor 之后，开始延迟即那次 monitor 的耗时。
- capsule 构建：腕部 224 286–288 ms，720 338–345 ms。
- 这次运行的 `original` 一组里，decompose 与 replan 解码每 token 20–41 ms，`fast` 一组 8–9 ms；随后同一时段的两段录制
  对照（本脚本与改动前的版本，都是 `original`）都是 7.6–9.2 ms，所以差别来自那段时间的共享 GPU 负载。
- 只记 L_reply 的旧版本（`original` 的决策时间记为回复结束）在同一对照里 monitor 中位 1.434 s，本版 L_decision 0.71 s、L_reply 1.435 s。

## 结果（Jetson Thor，单独运行，`1111_h15h16` 轮 H16）

官方 Qwen3.8-27B NVFP4，同样 6 段录制、66 次调用，GPU 空闲、经 GPU 锁运行（commit `7ed16ef9`）。monitor 中位（最大），单位 s。

| 组 | L_decision 224 | L_decision 720 | 开始延迟 224 | L_decision > 5.33 s | prefill 224 / 720 |
|---|---|---|---|---|---|
| `original` | 4.79（5.58） | 6.60（8.47） | 9.47（29.65） | 30 / 59 | — |
| `fast` | 2.63（2.79） | 4.38（4.64） | 0.00（1.03） | 0 / 59 | 2.03 / 3.69 |
| `fast` + `wy_lt` | 1.78（2.03） | 2.86（4.19） | 0.00（0.00） | 0 / 59 | 1.18 / 2.17 |

- `original` 的 monitor 有一半超过 5.33 s，后面的调用顺延，开始延迟逐次累积到 29.65 s。
- `fast` + `wy_lt`（`FLASHRT_QWEN36_THOR_LIN_PREFILL=parent`、`FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND=wy_lt`）：腕部 720 在 ep188 前几步仍有开始延迟（最大 6.05 s）；prefill 占 L_decision 的 66 %（224）/ 76 %（720）。
- 决策：各组与 5090 同一组逐调用相同 59 / 66；`fast` 与 `fast` + `wy_lt` 61 / 66。

## 共置一格（`run_colocation.sh`）

```
OUT=DIR PERIOD_MS=533 [ALONE=planner_process_thor_fast_wylt.jsonl] bash run_colocation.sh TAG {process|mps:P:E} {original|fast} CALLS_JSONL...
```

| 项 | 取值 |
|---|---|
| 执行器 | `../../../wam-inference-acceleration/experiments/colocation_5090/colocation_bench.py isolated`（ImageWAM fp8、10 步），独立进程，每 `PERIOD_MS` 释放一次 tick；`STOP_FILE` 让它在规划器结束后的第一次释放处停止，输出带计时窗口起点的墙钟 `t0_wall` |
| 规划器 | 本目录 `planner_process.py`，执行器计时窗口开始时启动（加载、热身、计时一遍） |
| 共享方式 | `process`：两个进程，不开 MPS；`mps:P:E`：本格自己的 MPS daemon，规划器 / 执行器的 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` 为 P / E |
| 对齐 | `colocation_align.py`：规划器每次调用按开始时刻与各阶段的同步计时切成 vit / restore / prefill / decode（capsule 构建单独一段），每个执行器 tick（释放 → 完成）归到重叠最长的阶段；只计规划器计时那一遍之内的 tick。输出各组 tick 数、p50 / p99 / max、超过各阈值的次数，以及规划器每类调用的 L_decision 与开始延迟（给 `ALONE` 时并列单独运行同一组录制的值） |

## 文件

| 文件 | 内容 |
|---|---|
| `planner_process.py` | 规划器进程 |
| `with_gpu_lock.sh` | 在 Thor 共用的 GPU 独占锁下运行一个命令 |
| `run_colocation.sh` | 一个共置格：执行器进程 + 规划器进程 + 对齐 |
| `colocation_align.py` | 执行器 tick 按规划器阶段分组，规划器调用按类型与腕部图分组 |
| `planner_process_thor_{original,fast,fast_wylt}.jsonl` | Thor 单独运行（`1111_h15h16` 轮 H16） |
| `planner_process_rtx_{fast,original}.jsonl` / `.log` | RTX 5090 功能检查，6 段录制（共享 GPU，见上） |
| `planner_process_rtx_{fast,original}.gpu_before` / `.gpu_after` | 两次运行前后 GPU 上的计算进程 |
