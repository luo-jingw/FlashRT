# Thor 测试清单

`0921x`/`0922`/`0922b`/`0922d`/`0922e`/`0922f`/`0922g`/`0922h`/`0922i` 把 X0/X6/X1/X4/X2/X3/X5/X7b/X8/X9/X10/X11/X12/X13 做完并落库（`THOR_STATUS_SUMMARY.md` 同名小节）。OPT-032 candidate 1（单流+双流）、candidate 3 第一、二轮接线、Phase 3（candidate 8）都在 Thor 上完全确认，`plan.md` Phase 1、Phase 2、Phase 3、Phase 4（第一、二轮）都已关闭。X13 确认了这轮大量改动 `pipeline_thor.py`/`checkpoint_loader.py`/`imagewam_thor.py` 之后默认路径**没有回归**（`0922i`，数字跟历史表基本重合）。`1021` 轮做完 X14、X15、X16（`THOR_STATUS_SUMMARY.md` 同名小节），已从本清单删除。`1028` 轮做完 Y0–Y4（Y5 未得到结果，见 issues.md ISSUE-099），已从本清单删除。`1029` 轮（Z0–Z2，短执行器配置与分母对齐）待测。剩下：candidate 3 Round 4（关闭双流/ActionDiT-double 边界，需要动 kernel 或改消费端设计）——等用户决定怎么走。RoboTwin 那张等它的 workload 声明，不在本清单。

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

### `1029` 轮：短执行器配置下的共享，与同 SM 数的分母对齐

问题：执行器为 ImageWAM nvfp4、2 步去噪、周期 50 / 100 ms 时，各共享方式下的错过率是否与 `1028` 轮（fp8、10 步、周期 260 ms）的结论一致？SM 分开以后，执行器相对"同样 SM 数单独运行"慢多少？只占带宽、不占 SM 的负载能让它慢多少？长上下文下的秒级停顿（issues.md ISSUE-098）落在规划器的哪个阶段？脚本在 `research/experiments/colocation_5090/`，命令都在该目录下运行；环境变量同 `1028` 轮，包括 `PYDEPS` / `PYTHONPATH`（transformers 5.17.0 的独立目录，`1028` 轮已装好，不重装）。整轮不改机器状态；GPU 上不要有别的任务，有的话记下来。全程 `NO_SMI=1`。

**Z0 环境，补 `1028` 轮没带回的字段。**

```
cd research/experiments/colocation_5090
export OUT=$HOME/thor_val/1029; mkdir -p $OUT
export FLASHRT_ROOT=$(git rev-parse --show-toplevel) DATASET_STATS_PATH=$(dirname $CKPT_PATH)/dataset_stats.json
export PYDEPS=$HOME/thor_val/pydeps_tf5; export PYTHONPATH=$PYDEPS:$PYTHONPATH NO_SMI=1
python -c "from flash_rt.hardware.jetson_clock_state import report_jetson_clock_state as r; r()" | tee $OUT/Z0_clock.log
git rev-parse HEAD | tee $OUT/Z0_commit.log
python -c "
import glob, json, os
for f in sorted(glob.glob(os.path.expanduser('~/thor_val/1028/cells/*.json'))):
    if f.endswith('.planner.json'): continue
    d = json.load(open(f)); print(os.path.basename(f)[:-5], d['green_sm'], d.get('planner_prefill_ms_p50'), d.get('planner_decode_ms_p50'))
for r in json.load(open(os.path.expanduser('~/thor_val/1028/Y2_planner_cost.json')))['rows']:
    print(r['images'], r['prompt_tokens'], round(r['prefill_ms']), round(r['decode_ms_per_token'], 1))
" | tee $OUT/Z0_1028_fields.log
```

带回：commit、`emc_locked`、`Z0_1028_fields.log` 全部行。判据：`green_sm` 给出 `1028` 轮 green25 / green50 的实际划分（推断是 8 / 12 与 16 / 4）；每 token 的解码 ms 给出这个规划器的解码速度，用来和其他规划器实现比较带宽需求。结论去向：`THOR_STATUS_SUMMARY.md` 的 `1028` 小节（补字段）。

**Z1 分母与纯带宽负载（约 15 分钟）。**

```
OUT=$OUT MPS=1 bash run_thor_align.sh probe | tee $OUT/Z1.log
```

带回：`Z1.log` 全部行。判据：
- `hog_probe_*` 行：纯拷贝负载在整卡、8 个 SM、16 个 SM 上能拿到的带宽（GB/s），是这块 Thor 带宽的实测上限；
- `probe_isolated` / `probe_greenalone25` / `probe_greenalone50` / `probe_mpsalone50`：执行器单独运行在 20 / 12 / 4 个 SM 上和 MPS 50 % 线程下的 p50 / p99，是 Z2 各格的分母；
- `probe_greenhog25` / `probe_greenhog50` 相对 `probe_greenalone25` / `probe_greenalone50` 慢多少，就是 SM 不相交时只靠带宽争用造成的减速；同一行的 `hog` 字段是执行器在旁边时负载实际拿到的带宽。

结论去向：同上，`1029` 小节。

**Z2 网格（约 1 小时）。**

```
OUT=$OUT MPS=1 bash run_thor_align.sh grid | tee $OUT/Z2.log
```

带回：`Z2.log` 全部行（每格一行：p50、p99、max、错过率、规划器调用次数、单次调用与解码耗时、`alloc`、`green`），出错的格带回它 `.log` 的最后 20 行；`$OUT/cells/p*_ctx16000_green25.json` 与 `p*_ctx16000_prio.json` 四个文件原样带回（它们含每次执行器调用的起止时间 `ticks_wall_s` 和每次规划器调用的起止与预填充耗时 `planner_records`）。判据：
- 周期 50 / 100 ms 下每种方式是否错过截止期，各格 p50 / p99 除以 Z1 里同 SM 数的分母（green25 用 `probe_greenalone25`，mps50 用 `probe_mpsalone50`，其余用 `probe_isolated`）；
- 同一方式在 50 与 100 ms 下错过率的差别，说明结果对周期余量有多敏感；
- 长上下文 green25 / prio 里超过周期的那几次执行器调用，落在规划器调用的哪一段（预填充：`t_start` 到 `t_start + prefill_ms`；解码：之后到 `t_end`；调用之间）；`alloc` 里 `segment.all.allocated` / `num_device_alloc` 非零，说明计时窗口内规划器在向驱动申请显存；
- `p50_isolated_end` / `p100_isolated_end` 与各自 start 的差值超过上面的效应，就不能当分母，要标明。

结论去向：同上；`research/experiments/colocation_5090/README.md` 的 Thor 一节；ISSUE-098 的结果写进 issues.md。

## 已完成的轮次（不再重跑）

逐轮结论已按"用法"第 2、3 条落库，不在本清单重复：每轮做了什么、数字是多少、口径是什么，看 `THOR_STATUS_SUMMARY.md` 的同名轮次小节（`eccf14f`、`a84916a`／`0919e`、`0920`、`0920s4`、`0920t`、`0920c`、`0921`、`0921x`、`0922`、`0922b`、`0922d`、`0922e`、`0922f`、`0922g`、`0922h`、`0922i`、`1021`、`1028`），各项结论看 `opportunities.md` 对应 OPT 条目。逐字的原始记录看 `git log`；本清单只保留"还没做"的东西。

---

## 已决定、不在这里测的

`text_trim` 转默认（原 E1）已经决定并落库：`default` profile 带 `text_trim=True`，新增 `native` profile；门禁与矩阵的默认口径跟着服务默认走（FA4 mot 与原生 VAE 进图后来也提升进了 `default`，`0921` 已在 Thor 上观察）；这些结论在 `plan.md` 的 "Decisions pending"、`opportunities.md` OPT-019/030 与 `THOR_STATUS_SUMMARY.md`。

三项 owner 决定（默认提升为最快配置、延迟基线按配置、标定 identity 加入相机几何）已落库，见 `plan.md` 的 "Decisions pending" 与 F1–F3；它们的 Thor 观察已在 `0921` 一轮完成。`plan.md` 的 "Open" 现在只剩目标工作负载自身未确认的声明（指令 token 数是否就是 128、checkpoint / 标定文件 / 图显存预算），它不是 Thor 测试项。

"Plan: ActionDiT small-M CUTLASS tile selection" 的 Thor 相位已在 `1021` 轮完成（X16）；`gemm_variant_autotune` 已按用户决定设为 nvfp4 的服务默认（`plan.md` "Decisions pending"）。
