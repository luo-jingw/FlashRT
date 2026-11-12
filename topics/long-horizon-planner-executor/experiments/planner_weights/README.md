# 27B 规划器权重：逐张量哈希比对

问题：联合测量里三份 `qwen38-27b-nvfp4-official/model.safetensors`（官方 Qwen3.8-27B-FP8 经
`research/topics/wam-inference-acceleration/experiments/qwen38_distill_bench/fp8_to_nvfp4_ct.py` 转换）是否逐张量相同：
RTX 5090 主机上的一份、Thor 上 `1104_h3` 轮转换、H8–H16 使用的一份（`$HOME/thor_val/ckpt/`）、Chenwei 在 Thor 上
B25 T0.7 转换的一份。

## 文件

| 文件 | 内容 |
|---|---|
| `hash_tensors.py` | 逐张量 sha256（只对张量字节，不含文件头），输出 `{name, dtype, shape, sha256}` 列表；与 Chenwei B25 `raw/b25/t0/hash_tensors.py` 逐字节相同 |
| `compare_tensor_hashes.py` | 比较两份哈希文件：共有 / 独有的张量数，dtype、shape、sha256 各有多少不同，不同的张量按名称模式分组 |
| `encode_variants_check.py` | 在 CPU 上用五种除法写法重新编码全部 NVFP4 张量，与已转换文件的码值逐个比较 |
| `encode_variants_rtx5090.tsv` | 上一项对 RTX 5090 主机那份的结果（每个张量、每种写法不同的码值数） |
| `tensors_sha256_rtx5090.json` | RTX 5090 主机那份的逐张量哈希（python 3.12.14、torch 2.14.0+cu130、safetensors 0.8.0） |

## 结果

| 比较 | 逐张量 | 整文件 sha256 |
|---|---|---|
| RTX 5090 主机 vs Chenwei Thor（B25 T0.7） | 1696 / 1696 相同；两个 JSON 逐字节相同（`7c018844…`） | `b168e1c7…` vs `3b8197ac…` |
| Thor H8 那份 vs 上两份 | 63 / 1696 个张量不同，全是 NVFP4 的 `weight_packed`；scale、global scale 与 bf16 张量全部相同 | `e05317eb…` |

- RTX 5090 那份与 Chenwei 那份的文件头长度相同（219 584 字节），`__metadata__` 的键顺序不同（format / source /
  conversion 对 conversion / format / source），两边 safetensors 0.8.0 / 0.7.0。整文件 sha256 的差别来自文件头写法（推断）。
- 两份源 FP8 的下载时间与版本不同（RTX 5090 主机 11-03；Chenwei 修订 `c3a462cc`，11-07），转换结果逐张量相同。

## 63 个张量为什么不同

`ct_nvfp4.encode` 两处除以 Python 标量（`amax / E2M1_MAX`、`scale / global_scale`），一处除以张量（`blocks / step`），
再用 `bucketize(right=False)` 按 E2M1 中点分档。FP8 源很粗，大量值恰好落在中点上，`step` 差一个 ulp 就会改变这些值的码。
`encode_variants_check.py` 的结果（相对 RTX 5090 那份的存储字节，码值不同的张量数，按 down / gate / up / o / k / q / v）：

| 写法 | 张量数 | 按类型 |
|---|---|---|
| 全部正确舍入的除法 | 116 | 32 / 26 / 26 / 8 / 10 / 7 / 7 |
| 除以标量用 float32 倒数相乘（两处） | **63** | **20 / 12 / 12 / 7 / 5 / 4 / 3** |
| 只有 `scale / global_scale` 用倒数 | 75 | 22 / 16 / 16 / 7 / 6 / 5 / 3 |
| 只有 `amax / E2M1_MAX` 用倒数 | 111 | 31 / 24 / 24 / 8 / 10 / 7 / 7 |
| 三处都用倒数 | 163 | 41 / 39 / 39 / 14 / 12 / 7 / 11 |

Thor H8 那份（`1104_h3` 轮在 Thor 上用 torch 2.9.1+cu130 转换）与 RTX 5090 那份的差别也是 63 个张量、20 / 12 / 12 / 7 / 5 / 4 / 3，
与第二行逐类相同；两份的 scale 相同，只有码值不同。RTX 5090 那份（torch 2.14）与 Chenwei 那份（Thor 上较新的 torch）相同。
推断：torch 2.9.1 在 CUDA 上把除以标量算成乘以 float32 倒数，较新的版本不是这样；中点上的值因此分到相邻的档。
逐张量名称的一致还没有比对（Thor 只带回了按名称模式的计数）。
