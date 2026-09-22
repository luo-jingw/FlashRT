# Thor 测试清单

`0921x`/`0922`/`0922b`/`0922d`/`0922e`/`0922f`/`0922g`/`0922h` 把 X0/X6/X1/X4/X2/X3/X5/X7b/X8/X9/X10/X11/X12 做完并落库（`THOR_STATUS_SUMMARY.md` 同名小节）。OPT-032 candidate 1（单流+双流）、candidate 3 第一、二轮接线、Phase 3（candidate 8）都在 Thor 上完全确认（`0922h`：`3 passed`，全部 `torch.equal`/`bit_exact=True`），`plan.md` Phase 1、Phase 2、Phase 3、Phase 4（第一、二轮）都已关闭。本节次待测项 X13：在决定 candidate 3 Round 4（关闭双流/ActionDiT-double 边界，需要动 kernel 或改消费端设计，还没开始）怎么做之前，先重新测一版"各精度"/"叠满配置"表，确认这轮大量改动 `pipeline_thor.py` 之后默认路径没有回归。RoboTwin 那张等它的 workload 声明，不在本清单。

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

### X13 重新测一版"各精度"+"叠满配置"表——本轮大量改动 `pipeline_thor.py`/`checkpoint_loader.py`/`imagewam_thor.py` 之后的默认路径回归确认
本轮（`fuse_qkv_norm_rope`/`fuse_res_norm_fp4`/`last_layer_kv_only` 三个新开关）全部是 opt-in、默认关闭，而且这三个开关目前都不在 `config_resolver.py` 的 `EXPERT_KEYS` 里，`imagewam_e2e_official_compare.py` 这条表**测不到它们**——这一项纯粹是"改了这么多 `pipeline_thor.py`，默认路径（`fuse_qkv_norm_rope=False` 等）的数字有没有被意外带偏"的回归确认，不是测新优化的收益。真要测新开关的收益，需要先把它们加进 `benchmarks/imagewam_fusion_ab.py` 的 `FLAGS`（这个我可以另外做，这次没做）。

直接跑 `scripts/imagewam_thor_validation.sh` 的第 3、4 步（跟"各精度"/"叠满配置"两张表当时的口径完全一样，`03_e2e_*_trim*`/`04_fid_*`/`04_e2e_goal_fp8_static_trim` 用的都是 `$FULL="N_TASKS=10 FRAMES=0,60 SEEDS=0,1"`）：
```
export STEPS="0 3 4"
export OUT=$HOME/thor_val/$(date +%m%d)
mkdir -p $OUT
bash scripts/imagewam_thor_validation.sh
```
不需要重编（这轮所有改动都是 Python）。前置的 `CKPT_PATH`/`FLUX2_SRC`/`BUNDLE` 等环境变量沿用已有设置（脚本文件头有完整清单，跟之前每一轮一样）。
判据：跟 `THOR_STATUS_SUMMARY.md` 里"各精度"（fp16 275.2ms/fp8_static 228.0ms/nvfp4 202.3ms 等）和"叠满配置"（nvfp4 106.1ms、fp8_static_cutlass 115.3ms）两张表的历史数字比——P50 应该在噪声范围内一致，`vs official`/`MAE vs GT` 应该跟历史值一致（cos 差异到小数点后 3-4 位算正常噪声，明显偏离才是回归）。把完整输出（尤其 `$OUT` 下 `03_e2e_*`、`04_fid_*`、`04_e2e_goal_fp8_static_trim` 各文件的关键行）带回来即可，不需要额外分析。
去向：THOR_STATUS_SUMMARY.md（更新"各精度"/"叠满配置"两张表）；如果发现真实回归，落 issues.md。

## 已完成的轮次（不再重跑）

逐轮结论已按"用法"第 2、3 条落库，不在本清单重复：每轮做了什么、数字是多少、口径是什么，看 `THOR_STATUS_SUMMARY.md` 的同名轮次小节（`eccf14f`、`a84916a`／`0919e`、`0920`、`0920s4`、`0920t`、`0920c`、`0921`、`0921x`、`0922`、`0922b`、`0922d`、`0922e`、`0922f`、`0922g`、`0922h`），各项结论看 `opportunities.md` 对应 OPT 条目。逐字的原始记录看 `git log`；本清单只保留"还没做"的东西。

---

## 已决定、不在这里测的

`text_trim` 转默认（原 E1）已经决定并落库：`default` profile 带 `text_trim=True`，新增 `native` profile；门禁与矩阵的默认口径跟着服务默认走（FA4 mot 与原生 VAE 进图后来也提升进了 `default`，`0921` 已在 Thor 上观察）；这些结论在 `plan.md` 的 "Decisions pending"、`opportunities.md` OPT-019/030 与 `THOR_STATUS_SUMMARY.md`。

三项 owner 决定（默认提升为最快配置、延迟基线按配置、标定 identity 加入相机几何）已落库，见 `plan.md` 的 "Decisions pending" 与 F1–F3；它们的 Thor 观察已在 `0921` 一轮完成。`plan.md` 的 "Open" 现在只剩目标工作负载自身未确认的声明（指令 token 数是否就是 128、checkpoint / 标定文件 / 图显存预算），它不是 Thor 测试项。

另一份 plan（"Plan: ActionDiT small-M CUTLASS tile selection"，roadmap item 1）还有一个面向 Thor 的相位是 `blocked`：`gemm_variant_autotune` 的 tile 扫描与 `infer()` A/B 从未在 Thor 上跑过（`benchmarks/imagewam_thor_small_m_tile_sweep.py`，`issues.md` ISSUE-023，`opportunities.md` OPT-018）。它不属于本清单的收尾范围（`fast`/`default` 都不开这个开关），列在这里只是为了不让人以为机器上的活已经全干完——如果要做，那是一个独立的短项。
