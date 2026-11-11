# Thor 测试清单

`0921x`/`0922`/`0922b`/`0922d`/`0922e`/`0922f`/`0922g`/`0922h`/`0922i` 把 X0/X6/X1/X4/X2/X3/X5/X7b/X8/X9/X10/X11/X12/X13 做完并落库（`THOR_STATUS_SUMMARY.md` 同名小节）。OPT-032 candidate 1（单流+双流）、candidate 3 第一、二轮接线、Phase 3（candidate 8）都在 Thor 上完全确认，`plan.md` Phase 1、Phase 2、Phase 3、Phase 4（第一、二轮）都已关闭。X13 确认了这轮大量改动 `pipeline_thor.py`/`checkpoint_loader.py`/`imagewam_thor.py` 之后默认路径**没有回归**（`0922i`，数字跟历史表基本重合）。`1021` 轮做完 X14、X15、X16（`THOR_STATUS_SUMMARY.md` 同名小节），已从本清单删除。`1028` 轮做完 Y0–Y4（Y5 未得到结果，见 issues.md ISSUE-099），已从本清单删除。`1029` / `1029b` / `1030` 轮做完，已从本清单删除。`1104_h` 轮做完 H1、H2，`1104_h3` 轮做完 H3a、H3，`1104_h3b` 轮做完 H3b，`1104_h3c` 轮做完 H3c，`1105_h6h4` 轮做完 H6、H4，`1110_h7h4bh8` 轮做完 H7、H4b、H8，`1111_h10` 轮做完 H10，`1111_h12h11h9` 轮做完 H12、H11、H9，`1111_h14h13` 轮做完 H14、H13，已从本清单删除。待测的是 H15、H15w（GPU 空闲时按 H12 的顺序重复，`native` 与 `wy_lt` 两条路径，把顺序、GPU 上的另一个进程与 Thor 前端自己的 prefill 路径分开）与 H16（规划器进程在 Thor 上单独运行，共置测量的基准）。另外剩下：candidate 3 Round 4（关闭双流/ActionDiT-double 边界，需要动 kernel 或改消费端设计）——等用户决定怎么走。RoboTwin 那张等它的 workload 声明，不在本清单。

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

三行依次做，合计约 2.5 小时；某一行失败不影响后面的行，带回失败信息后继续下一行。

GPU 锁：每个 GPU 命令都经 `$L`（`colocation_thor/with_gpu_lock.sh`）运行。它以只读方式打开 `/tmp/lingbot_va_gpu_exclusive.lock`，用 `flock(LOCK_EX)` 持有到命令结束，并打印等待、拿到、释放的时间。锁被共置测量的任务持有时会一直等，不要绕过，也不要停别人的进程。锁文件不存在时，命令照常运行，log 第一行会注明。

H15、H15w 的自变量是"GPU 上没有其他进程"。每行开始前看一次 `nvidia-smi`：拿到锁之后若仍有别人的计算进程（锁管不到的用户），照常跑，并在报告里写明是哪个进程、从何时到何时。

```
cd $FLASHRT_ROOT
git pull
export OUT=$HOME/thor_val/$(date +%m%d)_h15h16; mkdir -p $OUT
export CK=$HOME/thor_val/ckpt
export PYTHONPATH=$HOME/thor_val/pydeps_tf5:$PYTHONPATH
export PLANNER_CKPT=$CK/qwen38-27b-nvfp4-official PLANNER_MTP_DIR=$CK/qwen38-27b-fp8-official
D=research/topics/long-horizon-planner-executor/experiments
C=research/topics/wam-inference-acceleration/experiments/harness_replay_5090/calls
L=$D/colocation_thor/with_gpu_lock.sh
python -c "from flash_rt.hardware.jetson_clock_state import report_jetson_clock_state as r; r()" | tee $OUT/P0_clock.log
git rev-parse HEAD | tee $OUT/P0_commit.log
nvidia-smi | tee $OUT/P0_gpu.log
env | grep FLASHRT_QWEN36 | tee $OUT/P0_env.log
{ cat /etc/nv_tegra_release; nvidia-smi --query-gpu=driver_version --format=csv,noheader; ls /usr/local/cuda/lib64/libcublas.so.*; python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__)"; } 2>&1 | tee $OUT/P0_versions.log
```

`P0_env.log` 应为空（默认路径）。

### H15：GPU 空闲时按 H12 的顺序重复（`native` 路径）

问题：`1111_h12h11h9` 的 H12（`back_to_back` 顺序，GPU 上另有进程与 MPS）第 2–5 次与第 1 次 44 / 80 相同；`1111_h14h13` 的 H14（`after_previous` 顺序，GPU 空闲）79 / 80。两轮之间顺序与 GPU 上是否另有进程同时变了（issues.md ISSUE-114）。本行只把 GPU 状态换成空闲，顺序保持 H12 的。约 30 分钟。

```
nvidia-smi | tee $OUT/H15_gpu_start.log
bash $L python $D/monitor_call/reply_repeat.py thor back_to_back $OUT/reply_repeat_thor_official_idle.json $C/ep0_wrist224.jsonl $C/ep43_wrist224.jsonl 2>&1 | tee $OUT/H15_repeat.log
```

带回：`H15_repeat.log` 全部，`H15_gpu_start.log`；`reply_repeat_thor_official_idle.json`（复制到 `$D/monitor_call/` 并推送，附 `sha256sum`）。

判据：
- 每次调用 5 次里与第 1 次相同的次数（log 的 `N/5 identical`），与 H12（44 / 80，5 / 5 的调用 7 / 20）、H14（79 / 80，19 / 20）比较。
- 每次调用的第 2–5 次彼此是否相同。若只有前一次调用留下的状态起作用，预期每次调用的第 2–5 次彼此相同，第 1 次在部分调用里不同；若第 2–5 次中间也出现单次不同（如 H12 的 ep0 第 320 步），说明 GPU 空闲时也有别的来源。
- 与 H12 逐调用比：GPU 空闲时单次不同明显少于 H12，说明 H12 里的另一个进程（MPS）改变了规划器的回复，这正是共置时的条件。

### H15w：同一行，prefill 线性注意力走 `wy_lt`

问题：留存状态若来自 Thor 前端自己的 prefill 线性注意力路径（`_thor_lin_K_forward` 的 scratch 或状态槽），换到父类分块路径后应消失；5090 用父类路径，两种顺序都是 80 / 80。约 25 分钟。

```
export FLASHRT_QWEN36_THOR_LIN_PREFILL=parent FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND=wy_lt
env | grep FLASHRT_QWEN36 | tee $OUT/H15w_env.log
nvidia-smi | tee $OUT/H15w_gpu_start.log
bash $L python $D/monitor_call/reply_repeat.py thor back_to_back $OUT/reply_repeat_thor_official_idle_wylt.json $C/ep0_wrist224.jsonl $C/ep43_wrist224.jsonl 2>&1 | tee $OUT/H15w_repeat.log
unset FLASHRT_QWEN36_THOR_LIN_PREFILL FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND
```

带回：`H15w_env.log`、`H15w_gpu_start.log`、`H15w_repeat.log` 全部；`reply_repeat_thor_official_idle_wylt.json`（复制到 `$D/monitor_call/` 并推送，附 `sha256sum`）。

判据：
- 与 H15 同样的两个计数。H15 有第 1 次不同、H15w 为 80 / 80，说明留存状态在 Thor 前端的 prefill 线性注意力路径里；两行一样分散，说明来源在两条路径共用的部分（全注意力、GEMM、解码）。
- H15w 的回复与 H13 同一调用的参考回复（`$D/monitor_call/monitor_call_thor_official_wylt.json`）是否相同，只作参考。

结论去向（H15、H15w）：issues.md ISSUE-114；`THOR_STATUS_SUMMARY.md` 本轮小节；`$D/monitor_call/README.md` 运行间差异一节。

### H16：规划器进程在 Thor 上单独运行（6 段录制、66 次调用，三组）

问题：共置测量用 `colocation_thor/planner_process.py` 驱动 27B 规划器；它在 Thor 上还没跑过，720 腕部图的录制（5.7k token）也还没在 Thor 上跑过。本行在 Thor 上单独运行它（不开执行器），作为共置测量的"单独运行"基准：按 harness 节奏回放 6 段录制（ep0 / ep43 / ep188 × 腕部 224 / 720：6 次 decompose、59 次 monitor、1 次 replan），每次调用记开始、决策可用、结束。三组：`original`、`fast`、`fast` + `wy_lt` 路径。每组先不计时地热身一遍、再按节奏计时一遍，约 20–30 分钟，三组合计约 1.3 小时。

```
EPS="$C/ep0_wrist224.jsonl $C/ep0_wrist720.jsonl $C/ep43_wrist224.jsonl $C/ep43_wrist720.jsonl $C/ep188_wrist224.jsonl $C/ep188_wrist720.jsonl"
bash $L python $D/colocation_thor/planner_process.py thor original $OUT/planner_process_thor_original.jsonl $EPS 2>&1 | tee $OUT/H16_original.log
bash $L python $D/colocation_thor/planner_process.py thor fast $OUT/planner_process_thor_fast.jsonl $EPS 2>&1 | tee $OUT/H16_fast.log
export FLASHRT_QWEN36_THOR_LIN_PREFILL=parent FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND=wy_lt
env | grep FLASHRT_QWEN36 | tee $OUT/H16_env.log
bash $L python $D/colocation_thor/planner_process.py thor fast $OUT/planner_process_thor_fast_wylt.jsonl $EPS 2>&1 | tee $OUT/H16_fast_wylt.log
unset FLASHRT_QWEN36_THOR_LIN_PREFILL FLASHRT_QWEN36_TQ_PREFILL_GDN_BACKEND
nvidia-smi | tee $OUT/H16_gpu_end.log
```

带回：`H16_original.log`、`H16_fast.log`、`H16_fast_wylt.log`、`H16_env.log`、`H16_gpu_end.log` 全部；三个 jsonl（复制到 `$D/colocation_thor/` 并推送，附 `sha256sum`）。

判据：
- 完整性：每个 jsonl 66 行调用（`fast` 两组另有 6 行 `capsule`）；log 里有 `warm-up pass done`。
- 每组按调用类型（decompose / monitor / replan）与腕部图（224 / 720）：L_decision = `decision_s − start_s`、L_reply = `end_s − start_s` 的中位与最大值；monitor 开始时间相对到期时间的延迟（`start_s − due_s`）中位与最大值。`original` 的 monitor 若超过 5.33 s，后面的调用会顺延，延迟逐次累积。
- 决策：三组各自 59 次 monitor 的决策，与 5090 同一脚本（`$D/colocation_thor/planner_process_rtx_{original,fast}.jsonl`）逐调用比较；`fast_wylt` 与 `fast` 逐调用比较。
- `fast_wylt` 与 `fast`：腕部 224 的 monitor L_decision 应与 H13（1.77 s 与 2.61 s）同一量级；腕部 720 是新的数。

结论去向：`$D/colocation_thor/README.md` 的 Thor 一节；`THOR_STATUS_SUMMARY.md` 本轮小节；`research/topics/long-horizon-planner-executor/thor_rationale.md` W4。

## 已完成的轮次（不再重跑）

逐轮结论已按"用法"第 2、3 条落库，不在本清单重复：每轮做了什么、数字是多少、口径是什么，看 `THOR_STATUS_SUMMARY.md` 的同名轮次小节（`eccf14f`、`a84916a`／`0919e`、`0920`、`0920s4`、`0920t`、`0920c`、`0921`、`0921x`、`0922`、`0922b`、`0922d`、`0922e`、`0922f`、`0922g`、`0922h`、`0922i`、`1021`、`1028`、`1029`、`1029b`、`1030`、`1104_h`、`1104_h3`、`1104_h3b`、`1104_h3c`、`1105_h6h4`、`1110_h7h4bh8`、`1111_h10`、`1111_h12h11h9`、`1111_h14h13`），各项结论看 `opportunities.md` 对应 OPT 条目。逐字的原始记录看 `git log`；本清单只保留"还没做"的东西。

---

## 已决定、不在这里测的

`text_trim` 转默认（原 E1）已经决定并落库：`default` profile 带 `text_trim=True`，新增 `native` profile；门禁与矩阵的默认口径跟着服务默认走（FA4 mot 与原生 VAE 进图后来也提升进了 `default`，`0921` 已在 Thor 上观察）；这些结论在 `plan.md` 的 "Decisions pending"、`opportunities.md` OPT-019/030 与 `THOR_STATUS_SUMMARY.md`。

三项 owner 决定（默认提升为最快配置、延迟基线按配置、标定 identity 加入相机几何）已落库，见 `plan.md` 的 "Decisions pending" 与 F1–F3；它们的 Thor 观察已在 `0921` 一轮完成。`plan.md` 的 "Open" 现在只剩目标工作负载自身未确认的声明（指令 token 数是否就是 128、checkpoint / 标定文件 / 图显存预算），它不是 Thor 测试项。

"Plan: ActionDiT small-M CUTLASS tile selection" 的 Thor 相位已在 `1021` 轮完成（X16）；`gemm_variant_autotune` 已按用户决定设为 nvfp4 的服务默认（`plan.md` "Decisions pending"）。
