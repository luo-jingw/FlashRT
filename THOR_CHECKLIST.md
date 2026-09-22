# Thor 测试清单

`0921x`/`0922`/`0922b`/`0922d` 把 X0/X6/X1/X4/X2/X3/X5/X7b/X8 做完并落库（`THOR_STATUS_SUMMARY.md` 同名小节）。candidate 1 单流（backbone + ActionDiT）的接入代码在 Thor 上完全确认（`0922d`：`1 passed`，三处 `torch.equal`/`bit_exact=True`，`max_abs=0`），`plan.md` Phase 1 已关闭。candidate 1 双流（`_double_stream_layer`/`_action_double_layer`）已经写完并本机验证（backbone 与 ActionDiT 均 `torch.equal`，连续 3 次稳定），`plan.md` Phase 2 标记 completed，本节次只剩 X9 做 Thor 端确认。RoboTwin 那张等它的 workload 声明，不在本清单。

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

跑 native 门禁（`STEPS="8"`）与 S4-pipeline 的四行都要求下面三样已经在位，缺一样会**静默跳过而不是失败**：两个 native 测试文件在模块级 `pytest.importorskip("flash_rt.runtime.exec")`，`exec/` 不在时它们报 "2 skipped"、0 collected、退出码 0；parity gate 在 import 阶段就依赖 `exec/`。

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

## 待测

### X9 candidate 1（QKV拆分+QK-RMSNorm+RoPE 融合）双流接入（`_double_stream_layer`/`_action_double_layer`）的 Thor 位一致
本机（Ada）已经用 JIT 把真实 kernel 绑到 `flash_rt.flash_rt_kernels` 上，跑了实际提交的测试函数，backbone 双流层与 ActionDiT 双流层都 `torch.equal`，连续 3 次稳定。过程中抓到一个真实 bug，但是**测试自己构造权重时的 bug，不是接入代码的问题**：ActionDiT 双流层的 `proj.weight` 那个 `Fp16Linear` 的 `n`/`k` 两个参数传反了，导致 `key("proj.weight")` 跑出来的 GEMM 输出宽度不对，把 `action_proj_scratch` 缓冲区写越界（越界部分正好落在同一个 allocator 段里，`compute-sanitizer --tool memcheck` 因此报 0 错误，但 fused/unfused 各自越界覆盖的相邻内存不同，导致两次结果不一致）；靠对每个 `GemmRunner.fp16_nn` 调用逐个抓 `(M,N,K)` 和输出缓冲区做零拷贝快照、两边对比，才定位到这一个 GEMM。已修好并重新验证 3 次稳定。
```
python -m pytest tests/test_imagewam_thor_real_wiring.py -k fuse_qkv_norm_rope -q -s 2>&1 | tee $OUT/X9_wiring.log
```
判据：`torch.equal`（不是余弦）。这次 `-k fuse_qkv_norm_rope` 会连 X8 那个单流测试一起收集（两个测试都用这个关键字），预期都是通过；只关心新加的 `test_double_stream_and_action_double_fuse_qkv_norm_rope_bit_exact_at_real_shapes`。如果 Thor 上不过，把完整报错带回（尤其是不是同一处 `cuBLAS`/`illegal memory access`），不要重复本机已经做过的排查（种子、GemmRunner 重建、GEMM 形状逐个快照都已经做过）。
去向：opportunities.md OPT-032、plan.md Phase 2。

## 已完成的轮次（不再重跑）

逐轮结论已按"用法"第 2、3 条落库，不在本清单重复：每轮做了什么、数字是多少、口径是什么，看 `THOR_STATUS_SUMMARY.md` 的同名轮次小节（`eccf14f`、`a84916a`／`0919e`、`0920`、`0920s4`、`0920t`、`0920c`、`0921`、`0921x`、`0922`、`0922b`、`0922d`），各项结论看 `opportunities.md` 对应 OPT 条目。逐字的原始记录看 `git log`；本清单只保留"还没做"的东西。

---

## 已决定、不在这里测的

`text_trim` 转默认（原 E1）已经决定并落库：`default` profile 带 `text_trim=True`，新增 `native` profile；门禁与矩阵的默认口径跟着服务默认走（FA4 mot 与原生 VAE 进图后来也提升进了 `default`，`0921` 已在 Thor 上观察）；这些结论在 `plan.md` 的 "Decisions pending"、`opportunities.md` OPT-019/030 与 `THOR_STATUS_SUMMARY.md`。

三项 owner 决定（默认提升为最快配置、延迟基线按配置、标定 identity 加入相机几何）已落库，见 `plan.md` 的 "Decisions pending" 与 F1–F3；它们的 Thor 观察已在 `0921` 一轮完成。`plan.md` 的 "Open" 现在只剩目标工作负载自身未确认的声明（指令 token 数是否就是 128、checkpoint / 标定文件 / 图显存预算），它不是 Thor 测试项。

另一份 plan（"Plan: ActionDiT small-M CUTLASS tile selection"，roadmap item 1）还有一个面向 Thor 的相位是 `blocked`：`gemm_variant_autotune` 的 tile 扫描与 `infer()` A/B 从未在 Thor 上跑过（`benchmarks/imagewam_thor_small_m_tile_sweep.py`，`issues.md` ISSUE-023，`opportunities.md` OPT-018）。它不属于本清单的收尾范围（`fast`/`default` 都不开这个开关），列在这里只是为了不让人以为机器上的活已经全干完——如果要做，那是一个独立的短项。
