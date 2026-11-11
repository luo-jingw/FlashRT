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
| 输出 | 每次调用一行 JSON：到期 / 开始 / 决策可用 / 结束时刻，各阶段 ms，新 token 数，决策；capsule 构建单独一行 |

## 结果（RTX 5090，官方 Qwen3.8-27B，ep0 / ep43 腕部 224，各 1 次 decompose + 10 次 monitor）

功能检查。计时那一遍开始 10 s 后，另一个用户的进程开始占用同一块 GPU（约 5.7 GB，一直运行到结束），所以下表的耗时是**共享 GPU**
下的数，比独占时的 `../monitor_call/` 结果慢 15–25 %（prefill 400 / 318 ms，解码 10.5 / 9.1 ms 每 token）。

| 臂 | monitor 到决策可用中位（范围） | prefill 中位 | 解码中位 | 新 token 中位 | 最晚开始 | 非 `continue` |
|---|---|---|---|---|---|---|
| `fast` | 0.446 s（0.420–0.482） | 211 ms | 152 ms | 14 | +0.01 s | ep43 第 800 步 `replan` |
| `original` | 1.312 s（1.052–1.584） | 400 ms | 767 ms | 70.5 | +0.01 s | ep43 第 800 步 `replan` |

- 两臂决策相同（19 `continue`、1 `replan`），与 `../monitor_call/reply_repeat_rtx_official.json` 的 5090 决策相同。
- capsule 构建 285–289 ms；decompose 0.62–0.71 s。
- 不热身的第一版（同一脚本，GPU 独占）：前两次 monitor 的 prefill 518–668 ms、`original` 解码 8–26 ms 每 token，最长一次 3.73 s；
  热身之后范围收到 1.05–1.58 s。

## 文件

| 文件 | 内容 |
|---|---|
| `planner_process.py` | 规划器进程 |
| `planner_process_rtx_{fast,original}.jsonl` / `.log` | RTX 5090 功能检查（共享 GPU，见上） |
