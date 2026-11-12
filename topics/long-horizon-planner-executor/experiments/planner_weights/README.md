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
| `tensors_sha256_rtx5090.json` | RTX 5090 主机那份的逐张量哈希（python 3.12.14、torch 2.14.0+cu130、safetensors 0.8.0） |

## 结果

| 比较 | 逐张量 | 整文件 sha256 |
|---|---|---|
| RTX 5090 主机 vs Chenwei Thor（B25 T0.7） | 1696 / 1696 相同；两个 JSON 逐字节相同（`7c018844…`） | `b168e1c7…` vs `3b8197ac…` |
| Thor H8 那份 vs 上两份 | JSON sha256 `6efb6f80…`，不同；逐张量待比对 | `e05317eb…` |

- RTX 5090 那份与 Chenwei 那份的文件头长度相同（219 584 字节），`__metadata__` 的键顺序不同（format / source /
  conversion 对 conversion / format / source），两边 safetensors 0.8.0 / 0.7.0。整文件 sha256 的差别来自文件头写法（推断）。
- 两份源 FP8 的下载时间与版本不同（RTX 5090 主机 11-03；Chenwei 修订 `c3a462cc`，11-07），转换结果逐张量相同。
