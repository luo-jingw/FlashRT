# Thor 测试清单

本节次（配置整合 + `text_trim` 服务化）剩余的待测项。已完成的项已按"用法"第 2 条删除，结论在下面表格指定的文件里。

## 用法

1. `git pull` 后按顺序做。每项的命令、判据、结论去向都写在项内。
2. 做完一项：结论写进"结论去向"指定的文件，然后**从本清单删除这一项**。没有结论的中间产物不保留。
3. 每轮结束提交删减后的清单。历史看 `git log`，结论看下表的文件。

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

## R. ISSUE-085 修完后的回归（代码侧修完再跑）

`0919e` 把三个测试级现象拆成两个独立缺陷：FA4 dispatch 的 `capture_sync` 在录制中往捕获池分配（`beginAllocateToPool: already recording to mempool_id`），以及 AWQ 测试 plain 路径 kernel 计数为 0（AWQ 路径 285）；FA4 关的 graph-safety 是 4 passed（旧的红是 `capture_sync` 级联），FA4 开的 recover 仍红（fallback 吞掉 stand-in 后 `torch.equal` 失败），是另一个独立缺陷。三者都不影响服务路径（e2e `FA4 fallback=None`）。

```
python -m pytest tests/test_imagewam_fa4_dispatch.py -k capture_sync -q 2>&1 | tee $OUT/R_fa4_dispatch.log
python -m pytest tests/test_imagewam_awq.py -q 2>&1 | tee $OUT/R_awq.log
TRIM_PRECISION=nvfp4 TRIM_FA4=on python -m pytest tests/test_imagewam_text_trim_graph_safety.py -q -s 2>&1 | tee $OUT/R_fa4_recover.log
python -m pytest tests/test_imagewam_model_runtime_vae.py -q 2>&1 | tee $OUT/R_vae_ports.log
```
判据：四个文件全绿（`R_vae_ports.log` 是 ISSUE-086 的测试侧跟进，几何改成跟随 workload 后它期望的 token 数变了）。任一条仍红就写回 `issues.md` ISSUE-085。
去向：`issues.md` ISSUE-085（关闭或补充）。

---

## E. 收尾

### E1 `text_trim` 转默认（ISSUE-080 条件 2 与 native 一半）

六条条件里 1、3、4、5 的 ABI 一半、6 已完成；未完成的是条件 2 的 FA4 开分支（依赖上面 R 节）与条件 5 的 native 一半（C++，`plan.md` 的 S4 相位）。Python `infer()` 与 ABI 两条路径在 R 节转绿后即可把 `text_trim` 写进 `default` profile。这一条不是 Thor 测试，是 owner 改 `default` profile 的决定（`plan.md` 的 "Decisions pending"）。

### E2 fixture v2 的 manifest 带回入库

v2 的 fixture 已生成、trim gate 已过，只有 manifest 还没入库；Thor 上不推送，所以这一步是**把文件带回来**，不是在那儿提交。

在 Thor 上：

```
sha256sum tests/fixtures/imagewam_gate/imagewam_libero_gate_v2.manifest.json
```

把该文件的**内容**（约 6 KB JSON）贴回会话，或复制到双方都能取到的位置，连同上一步的 `sha256sum` 一起。

判据：有凭据的一方按原字节提交该文件，`sha256sum` 与 Thor 上的一致，并且 `tests/test_imagewam_regression_gate.py::test_committed_fixture_manifests_are_well_formed` 在有 GPU 与无 GPU 的机器上都过（它遍历 `tests/fixtures/imagewam_gate/*.manifest.json`）。数据目录继续留在 `$BUNDLE/imagewam_libero_gate_v2/`；若把它加进了 bundle，记得在 Thor 上重生成 `$BUNDLE/SHA256SUMS`。
去向：manifest 入 git；`issues.md` ISSUE-080 条件 4 的 Resolution 补一句"manifest 已入库"。
