# Thor 测试清单

本节次（配置整合 + `text_trim` 服务化）剩余的待测项。已完成的项已按"用法"第 2 条删除，结论在下面表格指定的文件里。

## 用法

1. `git pull` 后按顺序做。每项的命令、判据、结论去向都写在项内。
2. 做完一项：结论写进"结论去向"指定的文件（由有凭据的一方落库，见下），然后**从本清单删除这一项**。没有结论的中间产物不保留。
3. 每轮结束把"这一轮做了什么、数字是多少"作为报告带回；清单的删减由有凭据的一方完成。历史看 `git log`，结论看下表的文件。

**Thor 上不推送。** 这台机器没有 GitHub 凭据：只做 `git pull`/`checkout`，不做 `git commit`/`git push`，也不要指望 Thor 上的本地改动进入仓库。因此：

- 结论、数字、日志留在 `$OUT`，以报告形式（或把文件贴回/复制回）交给有凭据的一方入库；`$OUT` 与 `$BUNDLE` 都不进 git。
- 需要入库的文件（目前只有 v2 的 fixture manifest 一个）单独带回：把文件内容贴回来或复制回来，由有凭据的一方提交。带回时附 `sha256sum` 便于核对字节。
- 想在 Thor 上留备份可以本地 `git commit`，但那只是备份，**入库路径是"带回 + 由有凭据的一方提交"**，不是 Thor 上的 push。

原始日志只留在 `$OUT`，不进 git；只有回归基线进 git（`tests/fixtures/imagewam_gate/*.json`，fixture 数据本身留在 `$BUNDLE`）。

| 结论类型 | 写到 |
|---|---|
| 测得的数字与结论 | `opportunities.md` 对应 OPT 条目（OPT-016 … OPT-031） |
| ABI / native 服务路径的数字 | `opportunities.md` OPT-028（ABI）、OPT-029（native） |
| 默认值 / profile 决定、相位状态 | `plan.md` 的 "Execution status" 与 "Decisions pending" |
| 缺陷、未解释的现象、`text_trim` 转默认的条件 | `issues.md`（ISSUE-080、ISSUE-085、ISSUE-086 等） |
| 对外总结 | `THOR_STATUS_SUMMARY.md` |

## 前置（每轮一次）

环境变量与构建见 `scripts/imagewam_thor_validation.sh` 文件头。自上一轮起代码改动包含 VAE 几何（`vae_stage.py`、`vae_encoder.py`、前端），仍是 Python，**不含 kernel 与 C++ 改动，不需要重编**；ABI 行需要 `exec/`，native 行需要 `runtime/` 与 `flashrt_imagewam_native`。

```
export OUT=$HOME/thor_val/$(date +%m%d)
python -c "from flash_rt.hardware.jetson_clock_state import report_jetson_clock_state as r; r()" | tee $OUT/P0_clock.log
git rev-parse HEAD | tee $OUT/P0_commit.log
(cd $BUNDLE && sha256sum -c SHA256SUMS) | tee $OUT/P0_bundle.log
```

记录：commit hash；GPU 是否被其他进程占用；`emc_locked` 是否为 `null`（目前未锁定）。这三项写在本轮所有数字旁边。

---

## 已完成（不再重跑）

- `eccf14f`：LIBERO nvfp4 阶梯（spatial 三次重复、goal、libero_10）、叠满精度表、gate nvfp4 202.2 ms、A2 fp16（306.6 → 234.9 → 222.9 ms）、B2（AWQ 0.99970 / 真实 FP8 校准 0.99997）、E2 同口径基线与会话内极差（default 0.4 ms、stack 0.5 ms，`latency_baselines.json` 已按 202.2 ms 重定）、E3 fixture v2 与 trim gate（vs official 0.99931 / min 0.99898，P50 114.6 ms）、S1 三组 pytest、LIBERO 三路径（infer 202.3 / ABI 184.2 / native 183.8；`fast` 93.2 / 95.1 / 跳过）。
- `0919e`：A1 三项拆开定位、B1 两条 gate（e0m3 0.99781 / fp8_static_cutlass 0.99830，都过）、B3 每长度显存（首图 +218.0 MiB reserved / +206.3 allocated，之后 ~0；缓存 32 ≈ 221 MiB）、P 目标三路径（default 216.93 / 173.55 / 173.27；fast 137.64 / 139.35 / 跳过）与 LIBERO 保真逐值相同、S1 驱逐实测、E2 fixture v2 manifest。

数字与结论在 `opportunities.md`、`THOR_STATUS_SUMMARY.md`、`issues.md`（ISSUE-080/085/086）。

---
- `0920`：**R 四条全绿**（`capture_sync` 1 passed、AWQ 6 passed 且 `plain=285 awq=285`、FA4 开 graph-safety 4 passed 且恢复后 `equal=True cosine=1 max_abs=0`、`model_runtime_vae` 2 passed）→ ISSUE-085 已关闭，ISSUE-080 条件 2 满足。**E2 的 v2 manifest 已按原字节入库**（sha256 `a69a86ac…`，10954 字节，`test_committed_fixture_manifests_are_well_formed` 通过）。

---

## S4 native 多长度（本轮新增）

native model runtime 现在按长度带图（io config 声明长度表、`use_graph(key,exec)`、`set_text_length(key)` 在热路径选长度），所以**必须重编**：io config 变大、`use_graph` 换签名；旧 `.so` 会在加载时被 `native_library._check_layout` 拒掉（"config struct sizes differ from the ctypes mirror; rebuild the library"）。

```
# 0. 前置（commit/时钟/bundle）
cmake --build build -j --target flash_rt_kernels flash_rt_fp4 flashrt_imagewam_native
PB=$(python -c "import pybind11;print(pybind11.get_cmake_dir())")
cmake -S exec -B exec/build -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=$(which python) -Dpybind11_DIR=$PB && cmake --build exec/build -j
cmake -S runtime -B runtime/build -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=$(which python) -Dpybind11_DIR=$PB && cmake --build runtime/build -j
python -c "from flash_rt.models.imagewam.native_library import ImageWAMNativeLibrary as L; L()"   # 布局校验，静默即通过

IMAGEWAM_NATIVE_PRECISION=nvfp4 python -m pytest tests/test_imagewam_native_pipeline.py tests/test_imagewam_native_runtime.py -q -s 2>&1 | tee $OUT/S4_native.log
python tests/gate_imagewam_native_schema_parity.py --precision nvfp4 2>&1 | tee $OUT/S4_schema.log
python tests/gate_imagewam_native_parity.py --precision nvfp4 --graph python --bench-iters 50 2>&1 | tee $OUT/S4_parity_python.log
python tests/gate_imagewam_native_parity.py --precision nvfp4 --graph native --bench-iters 50 2>&1 | tee $OUT/S4_parity_native.log
```

判据：`test_native_tick_matches_infer_at_every_captured_length` 两个长度（`x0=6` 先、`14` 后）actions/actions_raw 都 `array_equal=True`，native manifest 的 `text_lengths={'default_key': 14, 'keys': [6, 14], 'per_prompt_length': True}`，`set_text_length` 对没有图的长度回 `-2`；schema gate PASS 且 `tests/data/imagewam_native_schema.records` **与 golden 逐行相同**（x0 不在记录里，不一致就是声明形状变了，要解释而不是重定基线）；parity gate 两种 `--graph` 的 tick 行全绿、六个 mutant 全检出、节点数与上一轮一致。**未裁剪（一 key）路径的数不能动**——`test_imagewam_native_pipeline.py` 整文件就是这条证据。P50 记进 OPT-029。
去向：`opportunities.md` OPT-029、`THOR_STATUS_SUMMARY.md`。

---

## E. 收尾

### E1 `text_trim` 转默认 —— 已决定（E1 = (c)）

`default` profile 现在带 `text_trim=True`，FA4 与原生 VAE 留在 `fast`；新增 `native` profile（不 trim、FA4 显式关、torch VAE 图外，内容等于旧的 `default`），给 S4 之前的 ABI/native 调用按名字切换，避免踩 R5。改 profile 是 plan 编辑，已记入 `plan.md` 的 "Decisions pending"。

配套已做/已记：矩阵脚本的开关行现在显式写 `TEXT_TRIM`（否则 `default` 行会悄悄变成 `vae_trim` 行，阶梯失去意义）；门禁自己的默认口径仍是"未裁剪参考配置"，要门禁服务默认就是 `--text-trim --manifest ...v2`（那条已经跑过并通过，P50 远低于未裁剪的延迟基线，不需要重测）。

**本轮没有新的 Thor 项。** S4（native 按长度带图）落地后要加的 Thor 项：native 多长度 tick 逐位一致，以及门禁默认口径若改成服务默认，再跑一次 `--text-trim --manifest ...v2` 记录数字。
