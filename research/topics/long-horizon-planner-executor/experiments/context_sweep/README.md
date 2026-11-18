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
