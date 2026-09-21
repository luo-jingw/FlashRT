# Thor 测试清单

上一节次（配置整合 + `text_trim` 服务化）的待测项已全部做完并按"用法"第 2 条删除。本节次（0921 跟进，`plan.md` 的 F1–F3）改了三件事，代码与 CPU 测试都在，需要 Thor 一轮：

1. **服务默认改为最快配置**：`default` profile = `text_trim` + FA4（backbone 与 mot 两个位点）+ 原生 VAE 进图。FA4 与 VAE 都是"auto"：机器或输入不满足时降级而不是报错（FA4 回退 cuBLAS 链路；没有 `ae_model_path` 就没有 VAE 阶段）。`fast` 是同一组开关的显式写法（缺 FA4 会报错）；`native` 不变，`default` 加 `consumer="native"` 解析成同一组。
2. **延迟基线按配置**：`latency_baselines.json` schema 2，`served_default` 未播种（`p50_ms: null`），`untrimmed_reference` 是原来的 202.2 ms。
3. **标定文件 identity 加入 `num_views`/`image_h`/`image_w`**：格式版本 3，只读版本 3；bundle 里两个标定文件读不了，要重录。

下面 N0–N5 按顺序做；**这一轮没有其他待测项**。

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

环境变量与构建的完整清单在 `scripts/imagewam_thor_validation.sh` 文件头。native 的 C++（pipeline 自己按长度录图）在 `0920t` 那一轮已经编过，此后没有新的 C++ 或 kernel 改动，提交都落在 Python 与记录文件上（`43c49ce`：长度表改成 property、native 的测试与门禁各自显式写 `use_fa4=False`；`73dc802`：一条 CPU 口径 pin）。所以本轮**不需要为了新代码重编**。

N5 里 S4-pipeline 的四行都要求下面三样已经在位，缺一样会**静默跳过而不是失败**：两个 native 测试文件在模块级 `pytest.importorskip("flash_rt.runtime.exec")`，`exec/` 不在时它们报 "2 skipped"、0 collected、退出码 0；parity gate 在 import 阶段就依赖 `exec/`。

- `flash_rt/libflashrt_imagewam_native.so`（本节第一条命令编出它；nvfp4 需要 cache 变量 `ENABLE_SM100_CUTLASS`）
- `exec/build`（ABI 面的 exec 层）
- `runtime/build`

parity gate 还需要这些环境变量，缺任何一个都是 `KeyError`（`tests/gate_imagewam_native_parity.py`）：`CKPT_PATH`、`FLUX2_SRC`、`FLUX2_AE_MODEL_PATH`（或 `AE_MODEL_PATH`）、`QWEN3_MODEL_SPEC`；schema gate 的 `CKPT_PATH` 是可选的（不给就用随机权重）。这些变量的取值见上面那个脚本的文件头。

```
export OUT=$HOME/thor_val/$(date +%m%d)
mkdir -p $OUT
python -c "from flash_rt.hardware.jetson_clock_state import report_jetson_clock_state as r; r()" | tee $OUT/P0_clock.log
git rev-parse HEAD | tee $OUT/P0_commit.log
(cd $BUNDLE && sha256sum -c SHA256SUMS) | tee $OUT/P0_bundle.log
```

`$BUNDLE` 由脚本文件头定义（示例 `BUNDLE=$HOME/thor_bundle`）；`$OUT` 每轮是一个带日期的**新**目录，所以先 `mkdir -p`，否则 `tee` 会因为目录不存在而失败、四条前置命令有三条不落日志。

记录：commit hash；GPU 是否被其他进程占用；`emc_locked` 是否为 `null`（目前未锁定）。这三项写在本轮所有数字旁边。

---

## 待测（0921 跟进：F1 默认提升、F2 基线、F3 标定身份）

### N0 重录两个标定文件（只有 `fp8_static*` 与 AWQ 的行需要；nvfp4 默认不读标定文件）
环境变量同 `benchmarks/imagewam_build_calibration.py` 文件头（`CKPT_PATH`、`FLUX2_AE_MODEL_PATH`、`FLUX2_SRC`、`QWEN3_MODEL_SPEC`、`DATA_ROOT`）。文件名沿用旧名（脚本里写死；`_v1`/`_v2` 是录制标签，不是格式版本）：
```
python benchmarks/imagewam_build_calibration.py --out $BUNDLE/imagewam_libero_calib_n64_v1.safetensors --n 64
python benchmarks/imagewam_build_calibration.py --out $BUNDLE/imagewam_libero_calib_n64_trim_v2.safetensors --n 64 --text-trim
(cd $BUNDLE && grep -v imagewam_libero_calib_n64 SHA256SUMS > SHA256SUMS.new && sha256sum imagewam_libero_calib_n64_v1.safetensors imagewam_libero_calib_n64_trim_v2.safetensors >> SHA256SUMS.new && mv SHA256SUMS.new SHA256SUMS)
```
判据：两个文件写出；`python -c "from flash_rt.models.imagewam.calibration_file import load_calibration as l; c=l('$BUNDLE/imagewam_libero_calib_n64_trim_v2.safetensors'); print(c.version, c.text_trim, {k: c.dims[k] for k in ('num_views','image_h','image_w')})"` 打印 `3 True {'num_views': 2, 'image_h': 224, 'image_w': 224}`（v1 文件是 `text_trim False`）。旧文件（版本 1/2）被拒并提示重录，这是预期行为。
去向：带回两个文件的 `sha256sum`，不进 git。

### N1 服务默认的 gate，并播种 `served_default` 基线
```
python tests/gate_imagewam_libero.py --precision nvfp4 --fixture-dir "$BUNDLE/imagewam_libero_gate_v2" --output-dir $OUT/N1_gate 2>&1 | tee $OUT/N1_gate.log
```
（不带任何 `--override`；manifest 默认就是 v2。）
判据：
- 打印的 `effective_config` 是 `precision=nvfp4 text_trim=True vae_encoder=native vae_graph=True use_fa4=True use_fa4_mot=True fa4_fallback_reason=None ...`；有回退就不是服务默认的数字，不要播种，把 `fa4_fallback_reason` 带回。
- fidelity 检查全过。fixture v2 的 `fp16` 参考是用普通 cuBLAS 链路 + torch VAE 录的，新默认跑的是 FA4 两个位点 + 原生 VAE，余弦可能略有偏移，要确认仍高于 nvfp4 阈值（`fidelity_thresholds.json`）；不过就是真问题，不要调阈值。
- 延迟检查 `ungated`（`served_default` 未播种），并打印 `latency_seed`。
播种：把 `$OUT/N1_gate/result.json` 里 `latency_seed.entry` 带回来，由有凭据的一方粘贴到 `latency_baselines.json` 的 `devices["thor"].baselines["nvfp4"]["served_default"]`。**同一命令重跑三次**，记录三次 P50 的极差（ISSUE-082 的教训：基线的 5% 余量要大于跑间波动）；GPU 是否独占、`emc_locked` 写在数字旁边。
再确认旧基线仍在管它自己的配置：
```
STEPS="2" OUT=$OUT/N1_ref BUNDLE=$BUNDLE bash scripts/imagewam_thor_validation.sh
```
判据：`02_gate_nvfp4` 通过，latency 检查用的是 `untrimmed_reference`（202.2 ms × 1.05）。
去向：`opportunities.md` OPT-030（数字）、`latency_baselines.json`（由有凭据的一方提交）、`THOR_STATUS_SUMMARY.md`。

### N2 提升默认带来的三个风险（各一次观察）
1. **FA4 首次调用编译**：在一个从未跑过 FA4 的进程里（或清掉 FA4 的编译缓存后），记录 `load_imagewam(profile="default")` 的构造耗时（冷）与紧接着的第二次构造耗时（热）。判据：冷启动多出的时间是一次性的、写下具体数字；热启动与改动前同量级。
2. **FA4 不可用时的降级**：`FLASHRT_THOR_FA4=0` 下跑 default，不应报错，`effective_config` 显示 `use_fa4=False use_fa4_mot=False`，P50 与 N3 的 `vae_trim` 行同量级。
   ```
   SUITE=libero_spatial PRECS=nvfp4 PROFILES="default" FLASHRT_THOR_FA4=0 TAG=n2_nofa4 bash scripts/imagewam_thor_matrix.sh
   ```
3. **进图 VAE 的固定输入**：目标工作负载（3 相机 256×256）上跑 default，确认 VAE 进图后 `infer()` 与 ABI 仍可用（native 行会被跳过：native 面拒绝进图 VAE，这是预期）：
   ```
   python benchmarks/imagewam_thor_path_bench.py --workload target --profile default 2>&1 | tee $OUT/N2_target_default.log
   ```
   判据：`infer`、`abi` 两行有数、与 `0919e` 的 216.93 / 173.55 ms 比较写出差；用 `--profile native` 再跑一次拿 native 行。
去向：OPT-019（FA4）、OPT-030、`THOR_STATUS_SUMMARY.md`。

### N3 配置矩阵（flag 行现在每行写明 VAE 与两个 FA4 位点）
```
SUITE=libero_spatial PRECS=nvfp4 bash scripts/imagewam_thor_matrix.sh
SUITE=libero_spatial PRECS=nvfp4 PROFILES="default fast native" bash scripts/imagewam_thor_matrix.sh
```
判据：
- 每行 `rc=0`，`FA4 bb`/`FA4 mot` 与行定义一致，`FA4 fallback` 为 `None`。
- `profile_default` 与 `profile_fast` 与 `stack` 行同一组开关：P50 在跑间波动内相同，vs official 不劣于 `stack`。
- `profile_native` 是无 FA4、无 VAE 进图的 trim 配置，记录数字。
- `default` 行（flag 行，未裁剪基线）约 202 ms、`vae` 约 190 ms，与 `0920t`/`eccf14f` 同量级；差得多就是行定义没生效，先看该行日志的 `effective_config`。
去向：OPT-030、`THOR_STATUS_SUMMARY.md` 的阶梯表。

### N4 标定重录后的 fp8_static 行（依赖 N0）
```
SUITE=libero_spatial PRECS=fp8_static_cutlass ROWS="stack" bash scripts/imagewam_thor_matrix.sh
```
判据：文件被接受（不报 identity 不符），vs official median 与 `c20f3a0` 的 0.99995 同量级。
去向：OPT-022。

### N5 pytest 与 ABI / native 回归
```
STEPS="1 8" OUT=$OUT/N5 BUNDLE=$BUNDLE bash scripts/imagewam_thor_validation.sh
```
（A1 的判据：先看 `01_pytest` 的第一处失败，别看总数；第一处若在 `test_imagewam_fa4_dispatch.py` 见下方"已知未决"。）
判据：`08_gate_abi`、`08_gate_abi_vae_graph`、`08_gate_native_schema`、`08_gate_native` 全过。运行时 identity 现在多了 `dims.num_views`、`dims.image_h`、`dims.image_w` 三项（只增不改）。
去向：OPT-028、OPT-029。

**已知未决**：本机（无 GPU）只验证了 CPU 测试与 stub 边界；`test_imagewam_fa4_dispatch.py` 的 `capture_sync` 模式是否污染后续 CUDA 状态（上一轮 96 failed/51 errors 的候选原因）在 Thor 上还没有单独查过，这一轮的 `01_pytest` 日志请保留第一处失败的完整报错。

---

## 已完成的轮次（不再重跑）

逐轮结论已按"用法"第 2、3 条落库，不在本清单重复：每轮做了什么、数字是多少、口径是什么，看 `THOR_STATUS_SUMMARY.md` 的同名轮次小节（`eccf14f`、`a84916a`／`0919e`、`0920`、`0920s4`、`0920t`、`0920c`），各项结论看 `opportunities.md` 对应 OPT 条目。逐字的原始记录看 `git log`；本清单只保留"还没做"的东西。

---

## 已决定、不在这里测的

`text_trim` 转默认（原 E1）已经决定并落库：`default` profile 带 `text_trim=True`，新增 `native` profile；门禁与矩阵的默认口径跟着服务默认走（FA4 mot 与原生 VAE 进图后来也提升进了 `default`，见 N 节）；这些结论在 `plan.md` 的 "Decisions pending"、`opportunities.md` OPT-019/030 与 `THOR_STATUS_SUMMARY.md`。

三项 owner 决定（默认提升为最快配置、延迟基线按配置、标定 identity 加入相机几何）已落库，见 `plan.md` 的 "Decisions pending" 与 F1–F3；它们的 Thor 观察就是上面的 N0–N5。`plan.md` 的 "Open" 现在只剩目标工作负载自身未确认的声明（指令 token 数是否就是 128、checkpoint / 标定文件 / 图显存预算），它不是 Thor 测试项。

另一份 plan（"Plan: ActionDiT small-M CUTLASS tile selection"，roadmap item 1）还有一个面向 Thor 的相位是 `blocked`：`gemm_variant_autotune` 的 tile 扫描与 `infer()` A/B 从未在 Thor 上跑过（`benchmarks/imagewam_thor_small_m_tile_sweep.py`，`issues.md` ISSUE-023，`opportunities.md` OPT-018）。它不属于本清单的收尾范围（`fast`/`default` 都不开这个开关），列在这里只是为了不让人以为机器上的活已经全干完——如果要做，那是一个独立的短项。
