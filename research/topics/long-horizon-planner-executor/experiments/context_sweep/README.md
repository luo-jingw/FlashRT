# 规划器耗时随上下文长度（Jetson Thor）

问题：上下文从 1k 增长到 128k token 时，规划器的 prefill 与 decode 各要多久，哪一段先超过检查间隔 Δ = 5.33 s。

## 结果（2026-11-18，Thor，GPU 独占）

| 上下文 token | prefill ms | decode ms/token | MTP 每次接受 |
|---|---|---|---|
| 1 024 | 534 | 29.8 | 2.91 |
| 2 048 | 1 118 | 28.6 | 3.71 |
| 4 096 | 2 285 | 29.9 | 2.91 |
| 8 192 | 4 858 | 34.5 | 2.94 |
| 16 384 | 10 870 | 33.0 | 3.32 |
| 32 768 | 26 358 | 35.4 | 3.16 |
| 65 536 | 71 216 | 97.1 | 3.08 |
| 131 071 | 216 541 | 101.3 | 2.98 |

- 整段 prefill 约 9k token 时单独超过 Δ；32k 以上增长快于线性。
- decode 在 32k 以内 29–35 ms/token，64k 以上约 100 ms/token，原因见 `issues.md` ISSUE-117。
- 合成提示词的 MTP 接受率高于真实 monitor 调用（约 1.8），真实回复的 decode 更慢。

## 开思考与关思考的整次调用（2026-11-18，Thor）

每个上下文长度 3 个规划器式问题（下一步做哪个子任务、某子任务出现几次、最近哪些被重规划），贪心解码到 EOS，上限 4 096 token；`FLASHRT_QWEN36_GRAPH_CACHE_MAX=8192`。整次调用 = prefill + decode，单位 s。

| 上下文 | 关思考（3 个问题） | 开思考（3 个问题） | 开思考的思考 token |
|---|---|---|---|
| 1k | 3.8 / 5.1 / 1.0 | 63.5 / 61.1 / 100.6 | 2 006 / 1 123 / 3 240 |
| 4k | 12.1 / 2.7 / 2.6 | ≥ 109.4 / 44.4 / ≥ 135.3 | ≥ 4 096 / 1 526 / ≥ 4 096 |
| 16k | 12.5 / 11.3 / 28.1 | ≥ 165.7 / 118.6 / ≥ 187.9 | ≥ 4 096 / 2 793 / ≥ 4 096 |
| 32k | 30.1 / 26.9 / 28.3 | ≥ 208.3 / 153.1 / 123.6 | ≥ 4 096 / 3 022 / 2 636 |

- 开思考时一次调用 44–208 s，12 次里 5 次到上限仍未想完；关思考时 4k 以上主要是 prefill。
- 解码 25–44 ms/token，随上下文变长。
- 默认的 CUDA graph 缓存上限（256）下，长解码会反复重录 graph，约 100 ms/token；同一组的第一次扫描（`out_thinking/`）就是这种情况，数字不可用。原因与验证见 `issues.md` ISSUE-117。

## 设置

| 项 | 取值 |
|---|---|
| 提交 | 4k–128k：`2eef0e26`；1k、2k：`fbe04169`（缓存至少按 8 448 token 分配，否则短提示词不走长上下文路径） |
| 规划器 | HF 官方 Qwen3.8-27B，NVFP4（`qwen38-27b-nvfp4-official-5090`），官方 MTP 头，FP8 KV，长上下文 agent 路径；`FLASHRT_QWEN36_THOR_LIN_PREFILL=parent`、`FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND=wy_lt` |
| 提示词 | 合成的机器人日志 + 一个要求详细总结的问题，按目标长度截断；关闭思考 |
| 每格 | 每个长度一个进程；热身一遍（prefill + 256 token decode，CUDA graph 按解码位置惰性录制），再计时一遍；decode 固定 256 token，不在 EOS 停 |

## 文件

| 文件 | 内容 |
|---|---|
| `context_sweep.py` | 一个或多个长度的 prefill / decode 计时 |
| `run_thor.sh` | Thor 上逐长度运行，经共享 GPU 锁 |
| `out/` | 每个长度的 JSON 与板子状态 |
| `thinking_sweep.py` / `run_thinking_thor.sh` | 开思考与关思考的整次调用 |
| `decode_probe.py` / `budget_probe.py` | 按解码位置、按 token 预算、按停止方式的解码计时 |
| `out_thinking_cache8192/` | 开思考扫描，graph 缓存上限 8 192；另有 64k 上下文扫描的复测 |
| `out_thinking/` | 开思考扫描，默认 graph 缓存（长解码被重录拖慢，只作对照） |
