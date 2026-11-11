# 线性注意力 prefill 各变体与逐 token 路径的差（RTX 5090 与 Jetson Thor）

问题：Thor 前端的 prefill 用逐步递归的 `native` 线性注意力（`Qwen36TorchFrontendThor._thor_lin_K_forward`），
一个 kernel 占 monitor prefill 的 55 %（`THOR_STATUS_SUMMARY.md` `1110_h7h4bh8` H4b）；5090 用 cuBLASLt 的 WY 分块
（`wy_lt`）。WY 是否在 SM110 上漂移，若漂移，在哪个变体上；不漂移的话 Thor 上能省多少 prefill。

## 结论

**Finding 1：`wy_lt` 在 Thor 上逐层与 5090 相同，不存在 SM110 特有的漂移。**
同一份层输入与递归状态上运行一层，与逐 token 路径比：Thor 上 out cos 中位 0.999765、最小 0.9994（第 50 层），
5090 上 0.999766 / 0.9994（第 50 层）。这是 WY 分块与逐步递归之间的算法差，5090 的默认路径带着同样的差。

**Finding 2：只在 Thor 上出错的是手写 mma 的 `wy` kernel。** 48 层里 39 层 cos < 0.99，最小 −0.71，递归状态出现 NaN；
5090 上同一变体与 `wy_lt` 相同。`wy_lt_f32gemm` 在两台机器上都在第 1 层出错（cos −0.37），与机器无关。

**Finding 3：Thor 上 `wy_lt` 的整段 prefill 2 294 ms，Thor 自己的路径 3 994 ms，少 1 700 ms（43 %）。**
每层 16.9 / 34.7 ms。`wy_lt` 在 Thor 上是 5090 的 7.0 倍（每层 16.9 / 2.43 ms）。

**Finding 4：最后一个位置的 logits 对逐层的微小差别极其敏感，不能用来判断 kernel 对错。**
逐层 cos ≥ 0.9994 的所有变体，整段 prefill 后与逐 token 路径的 logits cos 只有 0.57–0.72（两台机器上的两个 `native`
变体之间也在这一范围），第 0 个 token 的 top-1（`{"` 与 ```` ``` ````）随变体变化。判断换后端是否可用要看决策与回复，
不看 logits cos。

## 设置

| 项 | 取值 |
|---|---|
| 提示词 | H8 的 ep0 第 80 步 monitor 调用（腕部 224），4 093 token，两块 prefill：2 048 + 2 045 行 |
| 模型 | 官方 Qwen3.8-27B NVFP4，官方 FP8 MTP 头 |
| 前端 | 5090：`Qwen36VlTorchFrontendRtx`；Thor：`Qwen36VlTorchFrontendThor`；prefill 不录图（`FLASHRT_QWEN36_TQ_PREFILL_GRAPH=0`） |
| 参考 | `per_token`：每层逐行调用单 token 的 `_layer_forward_lin_nvfp4` |
| 逐层 | 用参考路径跑一遍整段 prefill，截下每个 K > 7 线性注意力层每一块的输入与进入时的递归 / 卷积状态；每个变体在同样的输入与状态上跑一层，与参考比较输出（cos、最大绝对差、相对 L2）、递归状态（cos）与耗时 |
| prompt 级 | 整段 prefill 里 K > 7 的线性注意力层换成该变体（全注意力层不变），比较最后一个位置的 logits；参考跑一次，其他热身一次、计时一次 |

变体：

| 名称 | 路径 |
|---|---|
| `own` | 前端自己的分派（Thor：`_thor_lin_K_forward`；5090：父类 K-row + `wy_lt`） |
| `thor_native` | `Qwen36TorchFrontendThor._thor_lin_K_forward`（5090 上先分配它的 scratch） |
| `rtx_native` | 父类 K-row 分块分支 + `native` 后端 |
| `wy` / `wy_lt` | 父类 K-row 分块分支 + 手写 mma 的 WY / cuBLASLt 的 WY |
| `wy_lt_f32gemm` / `wy_lt_f32state` | `wy_lt` 的两个 FP32 选项（`FLASHRT_QWEN36_TQ_PREFILL_GDN_FAST_CHUNK=f32` / `direct`） |

## 结果

chunk 0；chunk 1 的中位与最小值同一量级（`gdn_variant_drift_{rtx,thor}.log`）。

| 变体 | Thor out cos 中位 / 最小 | 5090 中位 / 最小 | Thor state cos 最小 | Thor ms / 层 | 5090 ms / 层 | Thor prefill ms | 5090 prefill ms | Thor top-1 = 参考 | 5090 top-1 = 参考 |
|---|---|---|---|---|---|---|---|---|---|
| `own` | 0.999998 / 0.9998 | 0.999766 / 0.9994 | 0.999899 | 34.7 | 2.43 | 3 994 | 321 | 是 | 否 |
| `thor_native` | 0.999998 / 0.9998 | 1.000000 / 0.9998 | 0.999899 | 34.7 | 14.2 | 3 993 | 1 448 | 是 | 是 |
| `rtx_native` | 0.999998 / 0.9998 | 1.000000 / 0.9998 | 0.999899 | 34.2 | 13.1 | 3 947 | 1 336 | 是 | 是 |
| `wy` | 0.9416 / −0.708 | 0.999766 / 0.9994 | NaN | 196.5 | 22.1 | 19 410 | 2 196 | 否 | 否 |
| `wy_lt` | 0.999765 / 0.9994 | 0.999766 / 0.9994 | 0.994067 | 16.9 | 2.43 | 2 294 | 322 | 是 | 否 |
| `wy_lt_f32gemm` | 0.9961 / −0.367 | 0.9961 / −0.369 | 0.994067 | 17.8 | 2.50 | 2 379 | 330 | 否 | 否 |
| `wy_lt_f32state` | 0.999764 / 0.9994 | 0.999765 / 0.9994 | 0.994064 | 18.8 | 2.61 | 2 475 | 340 | 否 | 否 |

- prompt 级 logits cos（对参考）：Thor `own` / `thor_native` 0.573，`rtx_native` 0.664，`wy_lt` 0.629；5090 `thor_native`
  0.696，`rtx_native` 0.723，`wy_lt` 0.635。
- 参考的 top-1 为 4754（`{"`）；5090 的 `wy_lt` 为 71093（```` ``` ````），与 5090 录制回复的第一个 token 相同。
- Thor 一行来自 `THOR_STATUS_SUMMARY.md` `1111_h12h11h9` 轮 H9（GPU 上另有一个进程存在，`emc_locked=null`）；ms 是同一轮里的相对值。

## 由此定下的改动

`FLASHRT_QWEN36_THOR_LIN_PREFILL=parent`（`flash_rt/frontends/torch/qwen36_thor.py`）让 Thor 前端 K > `_K_save_max`
的线性注意力层走父类 K-row 分块分支，后端由 `FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND` 决定（Thor 默认 `native`；
设为 `wy_lt` 即本实验的 `wy_lt` 变体）。默认仍为 `thor`。它在 monitor 调用上的决策、MTP 接受率与耗时由 Thor 行 H13 测量
（`opportunities.md` OPT-040 candidate 5）。

## 文件

| 文件 | 内容 |
|---|---|
| `gdn_variant_drift.py` | 探针：`python gdn_variant_drift.py {rtx|thor} OUT_JSON` |
| `run_rtx.sh` | RTX 5090 上的运行 |
| `gdn_variant_drift_rtx.json` / `.log` | RTX 5090 结果 |
| `gdn_variant_drift_thor.json` | Jetson Thor 结果（`1111_h12h11h9` 轮 H9） |
