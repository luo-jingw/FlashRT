# Thor 测试清单

`0921x`/`0922`/`0922b`/`0922d`/`0922e`/`0922f`/`0922g`/`0922h`/`0922i` 把 X0/X6/X1/X4/X2/X3/X5/X7b/X8/X9/X10/X11/X12/X13 做完并落库（`THOR_STATUS_SUMMARY.md` 同名小节）。OPT-032 candidate 1（单流+双流）、candidate 3 第一、二轮接线、Phase 3（candidate 8）都在 Thor 上完全确认，`plan.md` Phase 1、Phase 2、Phase 3、Phase 4（第一、二轮）都已关闭。X13 确认了这轮大量改动 `pipeline_thor.py`/`checkpoint_loader.py`/`imagewam_thor.py` 之后默认路径**没有回归**（`0922i`，数字跟历史表基本重合）。`1021` 轮做完 X14、X15、X16（`THOR_STATUS_SUMMARY.md` 同名小节），已从本清单删除。`1028` 轮（Y0–Y5，快慢模型共享 GPU）待测。剩下：candidate 3 Round 4（关闭双流/ActionDiT-double 边界，需要动 kernel 或改消费端设计）——等用户决定怎么走。RoboTwin 那张等它的 workload 声明，不在本清单。

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

### `1028` 轮：快慢模型共享 Thor 的 GPU

问题：ImageWAM 执行器与 Qwen3-VL-2B 规划器同在一块 Thor 上时，各共享方式下执行器的 p99 与截止期错过率、SM 划分手段是否可用、带宽争用有多大。RTX 5090 上的对照结果与脚本说明在 `research/experiments/colocation_5090/README.md`。脚本都在 `research/experiments/colocation_5090/`，以下命令都在该目录下运行，环境变量与 `scripts/imagewam_thor_validation.sh` 文件头相同（`run_thor.sh` 自己补 `FLASHRT_ROOT`、`DATASET_STATS_PATH`、`DROP_TEXT_ENCODER=1`）。整轮不改机器状态（不用 sudo、`nvpmodel -m`、`jetson_clocks`）；跑的时候 GPU 上不要有别的任务，有的话记下来。

**Y0 环境与规划器权重。**

```
cd research/experiments/colocation_5090
export OUT=$HOME/thor_val/1028; mkdir -p $OUT
export FLASHRT_ROOT=$(git rev-parse --show-toplevel) DATASET_STATS_PATH=$(dirname $CKPT_PATH)/dataset_stats.json
python -c "from flash_rt.hardware.jetson_clock_state import report_jetson_clock_state as r; r()" | tee $OUT/Y0_clock.log
git rev-parse HEAD | tee $OUT/Y0_commit.log
python -c "import transformers, cuda.bindings; from transformers import Qwen3VLForConditionalGeneration; print(transformers.__version__)" 2>&1 | tee $OUT/Y0_env.log
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --local-dir $FLASHRT_ROOT/tmp/checkpoints/Qwen3-VL-2B-Instruct 2>&1 | tail -2 | tee -a $OUT/Y0_env.log
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,clocks.sm,clocks.mem,power.draw --format=csv,noheader,nounits 2>&1 | tee $OUT/Y0_smi.log
nvidia-cuda-mps-control -d 2>&1 | tee $OUT/Y0_mps.log; echo get_server_list | nvidia-cuda-mps-control 2>&1 | tee -a $OUT/Y0_mps.log; echo quit | nvidia-cuda-mps-control
```

这两个导出在本轮所有命令里都要保持（新开 shell 要重新导出）。带回：commit、时钟状态（`emc_locked`）、transformers 版本、`cuda.bindings` 能否导入、权重是否下载成功、`nvidia-smi` 那一行的原样输出、MPS 守护进程能否启动。判据：`cuda.bindings` 或 `Qwen3VLForConditionalGeneration` 导入失败就停在这一项、带回报错（后面全部依赖它们）；`nvidia-smi` 那一行有任何字段不是数字，后面所有命令前加 `NO_SMI=1`；MPS 守护进程起不来，Y4 不加 `MPS=1`。结论去向：`THOR_STATUS_SUMMARY.md` 的 `1028` 小节。

**Y1 green context 能否在 Thor 上划分 SM。**

```
for n in 2 4 8; do python green_partition.py $n 2>&1 | tail -3; done | tee $OUT/Y1_green.log
```

带回：每次的 `device SMs`、两组各自的 SM 数、两组上 matmul 的耗时。判据：两组 matmul 耗时之比接近两组 SM 数的反比，说明 kernel 被限制在各自的 SM 组里；比值接近 1，说明没有限制住（Y3、Y4 里的 `green*` 格就没有意义，照跑但标明）。结论去向：同上。

**Y2 规划器单独运行的开销。**

```
python planner_qwen3vl.py sweep $OUT/Y2_planner_cost.json 2>&1 | grep -v -i warn | tail -20 | tee $OUT/Y2.log
```

带回：每行的上下文 token 数、预填充 ms、解码 ms / token、单次调用 ms、峰值显存。判据：与 5090 对照（259 token 0.30 s、16k token 0.79 s、32k token 1.55 s），给出 Thor 上的倍数。结论去向：同上。

**Y3 执行器单独运行，定控制周期。**

```
OUT=$OUT bash run_thor.sh probe | tee $OUT/Y3.log
```

带回：`probe_isolated`、`probe_greenalone25`、`probe_greenalone50` 三行的 p50 / p99。判据：`PERIOD_MS` 取 1.5 × `probe_isolated` 的 p99，向上取整到 10 ms，Y4 用这个值；两个 `greenalone` 行给出执行器只用 75% / 50% SM 时的基准。结论去向：同上。

**Y5 同一时间线上看执行器怎么被拖慢。**

```
PLANNER_CTX_EXTRA=16000 python overlap_trace.py $OUT/Y5_overlap.json 2>&1 | grep -v -i "warn\|autotune" | tail -25 | tee $OUT/Y5.log
```

带回：`K=` 那一行、`alone` 一行和 `overlap` 表的全部行。判据：看规划器预填充期间（`planner-cover` 接近 1 的行）执行器一次调用的 span 里 busy 与 wait 各占多少。5090 上是 span 307 ms，其中自身 kernel 34.7 ms、kernel 之间等待 272 ms；Thor 上 wait 占比低、busy 变大，说明主要矛盾从排队转成了带宽或算力争用。结论去向：同上。

**Y4 共享方式 × 规划器上下文的完整网格（最长，约 1–1.5 小时）。**

```
OUT=$OUT PERIOD_MS=<Y3 得到的值> MPS=<Y0 通过则为 1，否则为 0> bash run_thor.sh grid | tee $OUT/Y4.log
```

带回：`Y4.log` 的全部行（每格一行：p50、p99、max、错过率、规划器调用次数与单次耗时），以及 `$OUT/tegrastats_grid.log` 里每一格期间 EMC 利用率的平均值（按 `$OUT/cells/*.json` 里的时间段对齐，做不到就给整段的分布）。判据：
- 每种方式在短（约 260 token）、长（约 16k token）上下文下是否错过截止期；
- `green25` / `green50` 相对 `probe_greenalone25` / `probe_greenalone50` 慢了多少，这就是 SM 划分之后剩下的带宽争用（5090 上是 3–15%）；
- `grid_isolated_end` 与 `grid_isolated_start` 的差值超过上面任何一个效应，就说明这一轮漂移太大、不能当分母，要在报告里标明。

结论去向：同上，另外在 `research/experiments/colocation_5090/README.md` 新增一节 Thor 结果。

## 已完成的轮次（不再重跑）

逐轮结论已按"用法"第 2、3 条落库，不在本清单重复：每轮做了什么、数字是多少、口径是什么，看 `THOR_STATUS_SUMMARY.md` 的同名轮次小节（`eccf14f`、`a84916a`／`0919e`、`0920`、`0920s4`、`0920t`、`0920c`、`0921`、`0921x`、`0922`、`0922b`、`0922d`、`0922e`、`0922f`、`0922g`、`0922h`、`0922i`、`1021`），各项结论看 `opportunities.md` 对应 OPT 条目。逐字的原始记录看 `git log`；本清单只保留"还没做"的东西。

---

## 已决定、不在这里测的

`text_trim` 转默认（原 E1）已经决定并落库：`default` profile 带 `text_trim=True`，新增 `native` profile；门禁与矩阵的默认口径跟着服务默认走（FA4 mot 与原生 VAE 进图后来也提升进了 `default`，`0921` 已在 Thor 上观察）；这些结论在 `plan.md` 的 "Decisions pending"、`opportunities.md` OPT-019/030 与 `THOR_STATUS_SUMMARY.md`。

三项 owner 决定（默认提升为最快配置、延迟基线按配置、标定 identity 加入相机几何）已落库，见 `plan.md` 的 "Decisions pending" 与 F1–F3；它们的 Thor 观察已在 `0921` 一轮完成。`plan.md` 的 "Open" 现在只剩目标工作负载自身未确认的声明（指令 token 数是否就是 128、checkpoint / 标定文件 / 图显存预算），它不是 Thor 测试项。

"Plan: ActionDiT small-M CUTLASS tile selection" 的 Thor 相位已在 `1021` 轮完成（X16）；`gemm_variant_autotune` 已按用户决定设为 nvfp4 的服务默认（`plan.md` "Decisions pending"）。
