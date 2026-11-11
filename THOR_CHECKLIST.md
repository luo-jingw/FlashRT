# Thor 测试清单

`0921x`/`0922`/`0922b`/`0922d`/`0922e`/`0922f`/`0922g`/`0922h`/`0922i` 把 X0/X6/X1/X4/X2/X3/X5/X7b/X8/X9/X10/X11/X12/X13 做完并落库（`THOR_STATUS_SUMMARY.md` 同名小节）。OPT-032 candidate 1（单流+双流）、candidate 3 第一、二轮接线、Phase 3（candidate 8）都在 Thor 上完全确认，`plan.md` Phase 1、Phase 2、Phase 3、Phase 4（第一、二轮）都已关闭。X13 确认了这轮大量改动 `pipeline_thor.py`/`checkpoint_loader.py`/`imagewam_thor.py` 之后默认路径**没有回归**（`0922i`，数字跟历史表基本重合）。`1021` 轮做完 X14、X15、X16（`THOR_STATUS_SUMMARY.md` 同名小节），已从本清单删除。`1028` 轮做完 Y0–Y4（Y5 未得到结果，见 issues.md ISSUE-099），已从本清单删除。`1029` / `1029b` / `1030` 轮做完，已从本清单删除。`1104_h` 轮做完 H1、H2，`1104_h3` 轮做完 H3a、H3，`1104_h3b` 轮做完 H3b，`1104_h3c` 轮做完 H3c，`1105_h6h4` 轮做完 H6、H4，`1110_h7h4bh8` 轮做完 H7、H4b、H8，`1111_h10` 轮做完 H10，`1111_h12h11h9` 轮做完 H12、H11、H9，已从本清单删除。待测的是 H14（每次重复之前先生成上一个调用，回复是否变得可重复），之后 H13（prefill 线性注意力走分块路径之后的 monitor 网格）。另外剩下：candidate 3 Round 4（关闭双流/ActionDiT-double 边界，需要动 kernel 或改消费端设计）——等用户决定怎么走。RoboTwin 那张等它的 workload 声明，不在本清单。

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

开始前 GPU 上没有其他计算进程（有就等，不要停别人的进程）；若等不到，照常跑，并在报告里写明哪个进程在、从何时到何时。H14 约 30 分钟，H13 约 1 小时。

```
cd $FLASHRT_ROOT
git pull
export OUT=$HOME/thor_val/$(date +%m%d)_h14h13; mkdir -p $OUT
export CK=$HOME/thor_val/ckpt
export PYTHONPATH=$HOME/thor_val/pydeps_tf5:$PYTHONPATH
export PLANNER_CKPT=$CK/qwen38-27b-nvfp4-official PLANNER_MTP_DIR=$CK/qwen38-27b-fp8-official
D=research/topics/long-horizon-planner-executor/experiments
C=research/topics/wam-inference-acceleration/experiments/harness_replay_5090/calls
python -c "from flash_rt.hardware.jetson_clock_state import report_jetson_clock_state as r; r()" | tee $OUT/P0_clock.log
git rev-parse HEAD | tee $OUT/P0_commit.log
nvidia-smi | tee $OUT/P0_gpu.log
```

### H14：每次重复之前先生成上一个调用，回复是否变得可重复

问题：`1111_h12h11h9` 的 H12 里同一调用连续 5 次 `generate`，第 2–5 次与第 1 次只有 44 / 80 相同；7 次调用里第 2–5 次在同一位置偏离第 1 次，而第 1 次紧跟在上一个调用之后、第 2–5 次紧跟在同一提示词之后（issues.md ISSUE-114）。同一脚本换成 `after_previous` 顺序：每次重复之前先生成一次上一个调用（输出丢弃），让 5 次重复都有相同的前一次调用。默认路径，不设任何新环境变量。

```
python $D/monitor_call/reply_repeat.py thor after_previous $OUT/reply_repeat_thor_official_after_previous.json $C/ep0_wrist224.jsonl $C/ep43_wrist224.jsonl 2>&1 | tee $OUT/H14_repeat.log
```

带回：`H14_repeat.log` 全部；`reply_repeat_thor_official_after_previous.json`（复制到 `$D/monitor_call/` 并推送，附 `sha256sum`）。

判据：
- 每次调用 5 次里与第 1 次相同的次数（log 的 `N/5 identical`），与 H12（`$D/monitor_call/reply_repeat_thor_official.json`：5 / 5 的调用 7 / 20，第 2–5 次与第 1 次相同 44 / 80）比较。
- 若前一次调用留下的状态决定回复，预期大多数调用 5 / 5；若是调度相关的非确定性，预期与 H12 同样分散。第一个调用（ep0 第 80 步）没有前一个调用，与 H12 同样连续重复。

结论去向：issues.md ISSUE-114；`THOR_STATUS_SUMMARY.md` 本轮小节。

### H13：prefill 线性注意力走分块路径（`wy_lt`）之后的 monitor 网格

问题：`1111_h12h11h9` 的 H9 里，5090 的 WY 分块后端 `wy_lt` 在 Thor 上逐层与 5090 相同、不漂移，整段 prefill 2 294 ms，Thor 前端现在的逐步递归路径 3 994 ms（`$D/gdn_wy_thor/README.md`）。新开关 `FLASHRT_QWEN36_THOR_LIN_PREFILL=parent` 让 Thor 前端的 prefill 线性注意力层走父类分块分支，后端由 `FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND` 决定。用 H8 + H11 两组调用（47 次，含 3 次 `next`）跑同一网格，看耗时、MTP 接受率与决策。

```
export FLASHRT_QWEN36_THOR_LIN_PREFILL=parent FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND=wy_lt
env | grep FLASHRT_QWEN36 | tee $OUT/H13_env.log
python $D/monitor_call/monitor_call_bench.py thor own $OUT/monitor_call_thor_official_wylt.json $C/ep0_wrist224.jsonl $C/ep43_wrist224.jsonl $D/sim_closed_loop/calls/t*.jsonl 2>&1 | tee $OUT/H13_monitor_call.log
python $D/monitor_call/tables.py $OUT/monitor_call_thor_official_wylt.json | tee $OUT/H13_tables.md
unset FLASHRT_QWEN36_THOR_LIN_PREFILL FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND
```

带回：`H13_env.log`、`H13_monitor_call.log`、`H13_tables.md` 全部；`monitor_call_thor_official_wylt.json`（复制到 `$D/monitor_call/` 并推送，附 `sha256sum`）。任何一步失败：完整 traceback 与失败前最后 20 行。

判据：
- `H13_env.log` 里两个变量都在；无复用一格的 prefill 中位约 2.3 s 说明开关生效，约 4.0 s 说明没有生效（`native` 的 prefill 在 `1110_h7h4bh8` H4b / H8 与 `1111_h12h11h9` H11 三次为 3 981–3 994 ms）。
- 对齐复用 + 闭合即停一格：到决策可用中位与最大值，与 H8 的 2 608 ms、H11 的 2 684 ms 比较。
- 决策：各格 action 与同一运行的参考相同的次数；参考回复的决策与 `native` 运行（`$D/monitor_call/monitor_call_thor_official.json`、`$D/monitor_call/monitor_call_sim_thor_official.json`）逐调用比较，尤其 t0e1 第 240 步、t6e0 第 160 步、t6e1 第 160 步的 `next`。
- MTP：JSON 里每格的 `spec_attempts` / `spec_accepts`，与 `native` 运行比较（每次尝试接受的 token 数下降说明分块路径影响了草稿质量）。

结论去向：`opportunities.md` OPT-040 candidate 5；`$D/monitor_call/README.md` 的 Thor 一节；`THOR_STATUS_SUMMARY.md` 本轮小节。

## 已完成的轮次（不再重跑）

逐轮结论已按"用法"第 2、3 条落库，不在本清单重复：每轮做了什么、数字是多少、口径是什么，看 `THOR_STATUS_SUMMARY.md` 的同名轮次小节（`eccf14f`、`a84916a`／`0919e`、`0920`、`0920s4`、`0920t`、`0920c`、`0921`、`0921x`、`0922`、`0922b`、`0922d`、`0922e`、`0922f`、`0922g`、`0922h`、`0922i`、`1021`、`1028`、`1029`、`1029b`、`1030`、`1104_h`、`1104_h3`、`1104_h3b`、`1104_h3c`、`1105_h6h4`、`1110_h7h4bh8`、`1111_h10`、`1111_h12h11h9`），各项结论看 `opportunities.md` 对应 OPT 条目。逐字的原始记录看 `git log`；本清单只保留"还没做"的东西。

---

## 已决定、不在这里测的

`text_trim` 转默认（原 E1）已经决定并落库：`default` profile 带 `text_trim=True`，新增 `native` profile；门禁与矩阵的默认口径跟着服务默认走（FA4 mot 与原生 VAE 进图后来也提升进了 `default`，`0921` 已在 Thor 上观察）；这些结论在 `plan.md` 的 "Decisions pending"、`opportunities.md` OPT-019/030 与 `THOR_STATUS_SUMMARY.md`。

三项 owner 决定（默认提升为最快配置、延迟基线按配置、标定 identity 加入相机几何）已落库，见 `plan.md` 的 "Decisions pending" 与 F1–F3；它们的 Thor 观察已在 `0921` 一轮完成。`plan.md` 的 "Open" 现在只剩目标工作负载自身未确认的声明（指令 token 数是否就是 128、checkpoint / 标定文件 / 图显存预算），它不是 Thor 测试项。

"Plan: ActionDiT small-M CUTLASS tile selection" 的 Thor 相位已在 `1021` 轮完成（X16）；`gemm_variant_autotune` 已按用户决定设为 nvfp4 的服务默认（`plan.md` "Decisions pending"）。
