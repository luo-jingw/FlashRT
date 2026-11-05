# Thor 测试清单

`0921x`/`0922`/`0922b`/`0922d`/`0922e`/`0922f`/`0922g`/`0922h`/`0922i` 把 X0/X6/X1/X4/X2/X3/X5/X7b/X8/X9/X10/X11/X12/X13 做完并落库（`THOR_STATUS_SUMMARY.md` 同名小节）。OPT-032 candidate 1（单流+双流）、candidate 3 第一、二轮接线、Phase 3（candidate 8）都在 Thor 上完全确认，`plan.md` Phase 1、Phase 2、Phase 3、Phase 4（第一、二轮）都已关闭。X13 确认了这轮大量改动 `pipeline_thor.py`/`checkpoint_loader.py`/`imagewam_thor.py` 之后默认路径**没有回归**（`0922i`，数字跟历史表基本重合）。`1021` 轮做完 X14、X15、X16（`THOR_STATUS_SUMMARY.md` 同名小节），已从本清单删除。`1028` 轮做完 Y0–Y4（Y5 未得到结果，见 issues.md ISSUE-099），已从本清单删除。`1029` / `1029b` / `1030` 轮做完，已从本清单删除。`1104_h` 轮做完 H1、H2，`1104_h3` 轮做完 H3a、H3，`1104_h3b` 轮做完 H3b，`1104_h3c` 轮做完 H3c，已从本清单删除。待测的是 H6（解码每周期变慢的一段是否伴随拷贝带宽下降、空闲之后是否再出现）与 H4（monitor 长度 prefill 的逐 kernel 剖析），同一轮按 H6 → H4 的顺序做。另外剩下：candidate 3 Round 4（关闭双流/ActionDiT-double 边界，需要动 kernel 或改消费端设计）——等用户决定怎么走。RoboTwin 那张等它的 workload 声明，不在本清单。

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

### 本轮共同前置

H6 必须是 `git pull` 之后第一个用 GPU 的进程；开始前 GPU 上没有其他计算进程（有就等，不要停别人的进程）。全程约 30 分钟。

```
cd $FLASHRT_ROOT
git pull
export OUT=$HOME/thor_val/$(date +%m%d)_h6h4; mkdir -p $OUT
export CK=$HOME/thor_val/ckpt
export PYTHONPATH=$HOME/thor_val/pydeps_tf5:$PYTHONPATH
R=research/experiments/harness_replay_5090/thor
python -c "from flash_rt.hardware.jetson_clock_state import report_jetson_clock_state as r; r()" | tee $OUT/P0_clock.log
git rev-parse HEAD | tee $OUT/P0_commit.log
```

### H6：解码变慢的一段与拷贝带宽、空闲的关系

问题：`1104_h3c` 里同一形状的解码每周期在进程前 2.5 分钟是 136–137 ms，之后 87–88 ms，可读的 CPU 与 GPU 时钟都不变，prefill 只差 1.4 %（ISSUE-112）。（1）慢段里 1 GiB 设备内拷贝的带宽是否按同样比例下降；（2）稳态之后空闲 300 s，再开始时慢段是否再出现。

```
python $R/decode_cycle_drift.py thor 300 300 180 2>&1 | tee $OUT/H6_drift.log
python -I -c 'import json, sys; [print("%s %.0f s  %.1f ms/cycle  copy %.1f GB/s  prefill %.0f  cpu %d-%d kHz  %s" % (s["phase"], s["t_s"], s["ms_per_cycle"], s["copy_gbps"], s["prefill_ms"], min(s["cpu_khz"] or [-1]), max(s["cpu_khz"] or [-1]), s["devfreq_hz"])) for s in json.load(open(sys.argv[1]))["samples"]]' $OUT/decode_cycle_drift_thor.json | tee $OUT/H6_drift_table.txt
```

带回：`H6_drift.log` 的 `loaded:` 一行与 `idle 300 s` 一行；`H6_drift_table.txt` 全部。

判据：
- 拷贝带宽：慢段（ms / 周期 120 以上）与稳态（90 以下）的 `copy` 之比，与 ms / 周期之比（约 1.56）对照。比例相近：变慢在内存侧；拷贝不变：变慢不在纯带宽上。
- 空闲：`after` 段开头是否又出现 120 ms 以上的一段、持续多久。出现：慢段跟空闲之后的状态有关；不出现：跟进程开始有关。
- 5090 上同一脚本（`rtx 40 20 20`）：见 `research/experiments/harness_replay_5090/thor/decode_cycle_drift_rtx_probe_table.txt`。

结论去向：issues.md ISSUE-112；`THOR_STATUS_SUMMARY.md` 本轮小节；`research/experiments/harness_replay_5090/README.md` 的 Thor 每周期一段。

### H4：monitor 长度 prefill 的逐 kernel 剖析

问题：Thor 上 monitor 的 prefill（4 072 token）3.98 s，占一次调用的 61 %，按 2 × 27e9 FLOP / token 只合约 55 TFLOP/s（推算，OPT-040 candidate 5）。时间花在哪些 kernel 上：GEMM、注意力、线性注意力、量化各占多少；kernel 时间之和与 prefill 墙钟差多少（主机发射与空闲）；prefill graph 开与关的差别。

```
python $R/prefill_kernel_profile.py thor eager 2>&1 | tee $OUT/H4_eager.log
python $R/prefill_kernel_profile.py thor graph 2>&1 | tee $OUT/H4_graph.log
```

带回：两个日志的全部内容（`loaded:` 一行、prefill ms、kernel 数与时间和、类别表、前 40 个 kernel）。`$OUT/prefill_kernel_profile_thor_{eager,graph}.json` 留在 `$OUT`，不必带回。任何一步失败：完整 traceback 与失败前最后 20 行。

判据：
- 类别表里占比最大的一两类，以及前 40 个 kernel 里单个占比超过 5 % 的 kernel。`other` 一类若超过 10 %，带回的前 40 行里要能看出它是哪些 kernel。
- kernel 时间和与 prefill ms 之差：差得多，时间在主机侧（发射、同步、空闲）；差得少，时间在 kernel 本身。
- graph 与 eager 的 prefill ms 之差；graph 模式下若 kernel 数明显少于 eager（profiler 没有记到 graph 里的 kernel），以 eager 的类别表为准。
- 5090 上同一脚本：`research/experiments/harness_replay_5090/thor/prefill_kernel_profile_rtx_{eager,graph}.log`。

结论去向：`opportunities.md` OPT-040 candidate 5；`THOR_STATUS_SUMMARY.md` 本轮小节；`research/experiments/harness_replay_5090/README.md` 的 Thor 图像回放一节。

## 已完成的轮次（不再重跑）

逐轮结论已按"用法"第 2、3 条落库，不在本清单重复：每轮做了什么、数字是多少、口径是什么，看 `THOR_STATUS_SUMMARY.md` 的同名轮次小节（`eccf14f`、`a84916a`／`0919e`、`0920`、`0920s4`、`0920t`、`0920c`、`0921`、`0921x`、`0922`、`0922b`、`0922d`、`0922e`、`0922f`、`0922g`、`0922h`、`0922i`、`1021`、`1028`、`1029`、`1029b`、`1030`、`1104_h`、`1104_h3`、`1104_h3b`、`1104_h3c`），各项结论看 `opportunities.md` 对应 OPT 条目。逐字的原始记录看 `git log`；本清单只保留"还没做"的东西。

---

## 已决定、不在这里测的

`text_trim` 转默认（原 E1）已经决定并落库：`default` profile 带 `text_trim=True`，新增 `native` profile；门禁与矩阵的默认口径跟着服务默认走（FA4 mot 与原生 VAE 进图后来也提升进了 `default`，`0921` 已在 Thor 上观察）；这些结论在 `plan.md` 的 "Decisions pending"、`opportunities.md` OPT-019/030 与 `THOR_STATUS_SUMMARY.md`。

三项 owner 决定（默认提升为最快配置、延迟基线按配置、标定 identity 加入相机几何）已落库，见 `plan.md` 的 "Decisions pending" 与 F1–F3；它们的 Thor 观察已在 `0921` 一轮完成。`plan.md` 的 "Open" 现在只剩目标工作负载自身未确认的声明（指令 token 数是否就是 128、checkpoint / 标定文件 / 图显存预算），它不是 Thor 测试项。

"Plan: ActionDiT small-M CUTLASS tile selection" 的 Thor 相位已在 `1021` 轮完成（X16）；`gemm_variant_autotune` 已按用户决定设为 nvfp4 的服务默认（`plan.md` "Decisions pending"）。
