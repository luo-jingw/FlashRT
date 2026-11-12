# Thor 测试清单

`0921x`/`0922`/`0922b`/`0922d`/`0922e`/`0922f`/`0922g`/`0922h`/`0922i` 把 X0/X6/X1/X4/X2/X3/X5/X7b/X8/X9/X10/X11/X12/X13 做完并落库（`THOR_STATUS_SUMMARY.md` 同名小节）。OPT-032 candidate 1（单流+双流）、candidate 3 第一、二轮接线、Phase 3（candidate 8）都在 Thor 上完全确认，`plan.md` Phase 1、Phase 2、Phase 3、Phase 4（第一、二轮）都已关闭。X13 确认了这轮大量改动 `pipeline_thor.py`/`checkpoint_loader.py`/`imagewam_thor.py` 之后默认路径**没有回归**（`0922i`，数字跟历史表基本重合）。`1021` 轮做完 X14、X15、X16（`THOR_STATUS_SUMMARY.md` 同名小节），已从本清单删除。`1028` 轮做完 Y0–Y4（Y5 未得到结果，见 issues.md ISSUE-099），已从本清单删除。`1029` / `1029b` / `1030` 轮做完，已从本清单删除。`1104_h` 轮做完 H1、H2，`1104_h3` 轮做完 H3a、H3，`1104_h3b` 轮做完 H3b，`1104_h3c` 轮做完 H3c，`1105_h6h4` 轮做完 H6、H4，`1110_h7h4bh8` 轮做完 H7、H4b、H8，`1111_h10` 轮做完 H10，`1111_h12h11h9` 轮做完 H12、H11、H9，`1111_h14h13` 轮做完 H14、H13，`1111_h15h16` 轮做完 H15、H15w、H16，已从本清单删除。待测的是 H17（27B 规划器与执行器在 Thor 上共置，四种共享方式）。另外剩下：candidate 3 Round 4（关闭双流/ActionDiT-double 边界，需要动 kernel 或改消费端设计）——等用户决定怎么走。RoboTwin 那张等它的 workload 声明，不在本清单。

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

一行，六格依次做，合计约 50 分钟。某一格失败不影响后面的格，带回失败信息后继续下一格。

GPU 锁：每个 GPU 命令都经 `$L`（`colocation_thor/with_gpu_lock.sh`）运行。它以只读方式打开 `/tmp/lingbot_va_gpu_exclusive.lock`，用 `flock(LOCK_EX)` 持有到命令结束，并打印等待、拿到、释放的时间。锁被共置测量的任务持有时会一直等，不要绕过，也不要停别人的进程。锁文件不存在时，命令照常运行，log 第一行会注明。每格开始前看一次 `nvidia-smi`：拿到锁之后若仍有别人的计算进程，照常跑，并在报告里写明是哪个进程、从何时到何时。

执行器的环境变量（`CKPT_PATH`、`FLUX2_SRC`、`FLUX2_AE_MODEL_PATH`、`QWEN3_MODEL_SPEC`）与 `1104_h` 轮相同，取值见 `scripts/imagewam_thor_validation.sh` 文件头。

```
cd $FLASHRT_ROOT
git pull
export OUT=$HOME/thor_val/$(date +%m%d)_h17; mkdir -p $OUT
export CK=$HOME/thor_val/ckpt
export PYTHONPATH=$HOME/thor_val/pydeps_tf5:$PYTHONPATH
export PLANNER_CKPT=$CK/qwen38-27b-nvfp4-official PLANNER_MTP_DIR=$CK/qwen38-27b-fp8-official
export FLASHRT_ROOT=$PWD DATASET_STATS_PATH=$(dirname $CKPT_PATH)/dataset_stats.json
D=research/topics/long-horizon-planner-executor/experiments
C=research/topics/wam-inference-acceleration/experiments/harness_replay_5090/calls
W=research/topics/wam-inference-acceleration/experiments/colocation_5090
L=$D/colocation_thor/with_gpu_lock.sh
python -c "from flash_rt.hardware.jetson_clock_state import report_jetson_clock_state as r; r()" | tee $OUT/P0_clock.log
git rev-parse HEAD | tee $OUT/P0_commit.log
nvidia-smi | tee $OUT/P0_gpu.log
nvidia-smi -q 2>&1 | grep -i -B1 -A3 mig | tee $OUT/P0_mig.log
free -g | tee $OUT/P0_mem.log
env | grep FLASHRT_QWEN36 | tee $OUT/P0_env.log
```

`P0_env.log` 应为空；`P0_mig.log` 只读查询，记下 Thor 是否报告 MIG 模式（不要尝试开启）。

### H17：27B 规划器与执行器在 Thor 上共置，四种共享方式

问题：Thor 上把两个模型按 SM 隔开，执行器是否仍受规划器影响、规划器又慢多少。此前的数来自 2B / 7B 规划器（`1028`–`1030`、`1104_h` 轮，Chenwei C-#9 / C-#12），27B 没有测过。三个要读的量：
- 与规划器 decode 重叠的执行器 tick 是否仍变慢（`1030` 轮 V2：2B 替身在 SM 分区下 1.3–1.7 倍，推测来自共享的内存带宽）；
- 快路径的 prefill 只剩几百个 token，长 prefill 末尾的秒级停顿（ISSUE-098）是否还出现；
- 规划器限在约 4 个 SM 时，L_decision 比单独运行（H16 `fast_wylt`）慢多少，腕部 720 是否超过 Δ = 5.33 s。

设计：执行器（ImageWAM fp8、10 步，`colocation_bench.py isolated`，每 533 ms 释放一次 tick）与规划器进程（`planner_process.py`，`fast` + `wy_lt`，回放 ep0 腕部 224 / 720 两段录制：2 次 decompose、约 20 次 monitor）各是一个进程，`colocation_thor/run_colocation.sh` 在一格里先启动执行器、到它的计时窗口开始时再启动规划器，规划器结束后让执行器停止，最后用 `colocation_align.py` 在墙钟上把每个 tick 归到它重叠最长的规划器阶段（vit / prefill / decode / capsule / 无）。只统计规划器计时那一遍之内的 tick。

| 格 | 共享方式 | 规划器 / 执行器 MPS 线程上限 |
|---|---|---|
| H17a | 执行器单独，60 s | — |
| H17b | 两个进程，不开 MPS（驱动分时） | — |
| H17c | MPS，不限 | 100 % / 100 % |
| H17d | MPS，对半 | 50 % / 50 % |
| H17e | MPS，规划器约 4 个 SM | 20 % / 80 % |
| H17f | 执行器单独，60 s（漂移） | — |

```
export FLASHRT_QWEN36_THOR_LIN_PREFILL=parent FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND=wy_lt
env | grep FLASHRT_QWEN36 | tee $OUT/H17_env.log
EPS="$C/ep0_wrist224.jsonl $C/ep0_wrist720.jsonl"
export PERIOD_MS=533 ALONE=$PWD/$D/colocation_thor/planner_process_thor_fast_wylt.jsonl
R=$D/colocation_thor/run_colocation.sh
mkdir -p $OUT/H17a $OUT/H17f
bash $L bash -c "cd $W && DROP_TEXT_ENCODER=1 NO_SMI=1 python colocation_bench.py isolated 0 60 $OUT/H17a/exec.json" 2>&1 | tee $OUT/H17a.log
bash $L bash $R H17b process fast $EPS 2>&1 | tee $OUT/H17b.log
bash $L bash $R H17c mps:100:100 fast $EPS 2>&1 | tee $OUT/H17c.log
bash $L bash $R H17d mps:50:50 fast $EPS 2>&1 | tee $OUT/H17d.log
bash $L bash $R H17e mps:20:80 fast $EPS 2>&1 | tee $OUT/H17e.log
bash $L bash -c "cd $W && DROP_TEXT_ENCODER=1 NO_SMI=1 python colocation_bench.py isolated 0 60 $OUT/H17f/exec.json" 2>&1 | tee $OUT/H17f.log
unset FLASHRT_QWEN36_THOR_LIN_PREFILL FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND
nvidia-smi | tee $OUT/H17_gpu_end.log
```

带回：`H17_env.log`、`H17a.log` … `H17f.log`、`H17_gpu_end.log` 全部；每格目录下的 `align.json`、`exec.json`、`planner.jsonl`、`exec.log`、`planner.log`（复制到 `$D/colocation_thor/h17/<格>/` 并推送，附 `sha256sum`；`tegrastats.log` 只带回整段平均的 `VDD_GPU` 与 `RAM`）。

判据：
- 完整性：每个共置格的 `planner.log` 有 `warm-up pass done`、`planner rc=0 executor rc=0`；`exec.json` 的 `stopped_by_file` 为 `true`；`colocation_align.py` 打印 `executor window covers it: True`。MPS 格的 log 开头有 `CUDA_MPS_PIPE_DIRECTORY`。两个模型同时加载时若内存不足，带回报错与 `free -g`。
- 漂移：H17a 与 H17f 的 p50 / p99。共置格之间小于这个漂移的差别不读。
- 执行器（`colocation_align.py` 的阶段表）：每格 `decode`、`prefill`、`capsule`、`none` 各组的 p50 / p99 / max 与 `>200` / `>533` 计数。`decode` 组 p50 与 H17a p50 之比在 H17d、H17e（SM 受限）里是否仍明显大于 1；`prefill` 与 `capsule` 组有没有秒级的 max。
- 规划器（阶段表下面的调用表）：monitor 的 L_decision 中位 / 最大按腕部与括号里的单独运行比较；H17e 里腕部 720 的 L_decision 是否超过 5.33 s，`start-due` 是否随调用增长（排队）。

结论去向：`$D/colocation_thor/README.md` 新增 Thor 共置一节；`THOR_STATUS_SUMMARY.md` 本轮小节；issues.md ISSUE-098（短 prefill 下停顿是否仍在）；`research/topics/long-horizon-planner-executor/summary_1111.html` Q2c。

## 已完成的轮次（不再重跑）

逐轮结论已按"用法"第 2、3 条落库，不在本清单重复：每轮做了什么、数字是多少、口径是什么，看 `THOR_STATUS_SUMMARY.md` 的同名轮次小节（`eccf14f`、`a84916a`／`0919e`、`0920`、`0920s4`、`0920t`、`0920c`、`0921`、`0921x`、`0922`、`0922b`、`0922d`、`0922e`、`0922f`、`0922g`、`0922h`、`0922i`、`1021`、`1028`、`1029`、`1029b`、`1030`、`1104_h`、`1104_h3`、`1104_h3b`、`1104_h3c`、`1105_h6h4`、`1110_h7h4bh8`、`1111_h10`、`1111_h12h11h9`、`1111_h14h13`、`1111_h15h16`），各项结论看 `opportunities.md` 对应 OPT 条目。逐字的原始记录看 `git log`；本清单只保留"还没做"的东西。

---

## 已决定、不在这里测的

`text_trim` 转默认（原 E1）已经决定并落库：`default` profile 带 `text_trim=True`，新增 `native` profile；门禁与矩阵的默认口径跟着服务默认走（FA4 mot 与原生 VAE 进图后来也提升进了 `default`，`0921` 已在 Thor 上观察）；这些结论在 `plan.md` 的 "Decisions pending"、`opportunities.md` OPT-019/030 与 `THOR_STATUS_SUMMARY.md`。

三项 owner 决定（默认提升为最快配置、延迟基线按配置、标定 identity 加入相机几何）已落库，见 `plan.md` 的 "Decisions pending" 与 F1–F3；它们的 Thor 观察已在 `0921` 一轮完成。`plan.md` 的 "Open" 现在只剩目标工作负载自身未确认的声明（指令 token 数是否就是 128、checkpoint / 标定文件 / 图显存预算），它不是 Thor 测试项。

"Plan: ActionDiT small-M CUTLASS tile selection" 的 Thor 相位已在 `1021` 轮完成（X16）；`gemm_variant_autotune` 已按用户决定设为 nvfp4 的服务默认（`plan.md` "Decisions pending"）。
