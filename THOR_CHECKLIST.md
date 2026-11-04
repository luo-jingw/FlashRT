# Thor 测试清单

`0921x`/`0922`/`0922b`/`0922d`/`0922e`/`0922f`/`0922g`/`0922h`/`0922i` 把 X0/X6/X1/X4/X2/X3/X5/X7b/X8/X9/X10/X11/X12/X13 做完并落库（`THOR_STATUS_SUMMARY.md` 同名小节）。OPT-032 candidate 1（单流+双流）、candidate 3 第一、二轮接线、Phase 3（candidate 8）都在 Thor 上完全确认，`plan.md` Phase 1、Phase 2、Phase 3、Phase 4（第一、二轮）都已关闭。X13 确认了这轮大量改动 `pipeline_thor.py`/`checkpoint_loader.py`/`imagewam_thor.py` 之后默认路径**没有回归**（`0922i`，数字跟历史表基本重合）。`1021` 轮做完 X14、X15、X16（`THOR_STATUS_SUMMARY.md` 同名小节），已从本清单删除。`1028` 轮做完 Y0–Y4（Y5 未得到结果，见 issues.md ISSUE-099），已从本清单删除。`1029` / `1029b` / `1030` 轮做完，已从本清单删除。剩下：candidate 3 Round 4（关闭双流/ActionDiT-double 边界，需要动 kernel 或改消费端设计）——等用户决定怎么走。RoboTwin 那张等它的 workload 声明，不在本清单。

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

H1、H2 只用 origin 上已有的代码（`9cb3511` 起即可），不依赖本地未推送的提交。H3 依赖 27B 规划器的代码与权重，要等这些提交推送之后。三项互不依赖。

环境同 `1028` 轮（`run_thor.sh` 文件头）：`CKPT_PATH`、`FLUX2_SRC`、`FLUX2_AE_MODEL_PATH`、`QWEN3_MODEL_SPEC`、`DATASET_STATS_PATH`；规划器替身 Qwen3-VL-2B 需要 `$HOME/thor_val/pydeps_tf5` 前置在 `PYTHONPATH`；`NO_SMI=1`。

### H2：执行器换指令（文本编码器常驻）的耗时与显存

问题：Thor 是 128 GB 统一内存，执行器能否让 Qwen3-4B 文本编码器常驻，换一次子任务指令要多久。5090 上编码器常驻时换指令 30 ms（新的上下文长度 240 ms，进程内第一次 751 ms），但规划器常驻时编码器放不下。

```
export OUT=$HOME/thor_val/$(date +%m%d)_h; mkdir -p $OUT/cells
cd $FLASHRT_ROOT/research/experiments/colocation_5090
free -g | tee $OUT/H2_free.log
python - <<'EOF' 2>&1 | tee $OUT/H2_instruction_switch.log
import os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.environ["FLASHRT_ROOT"], "benchmarks"))
from _imagewam_libero_frames import center_crop_resize
import flash_rt.frontends.torch.imagewam_thor as F
from flash_rt.models.imagewam.workload import ImageWAMWorkload
fe = F.load_imagewam(os.environ["CKPT_PATH"], ImageWAMWorkload.libero(), precision="fp8",
                     ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"], flux2_src=os.environ["FLUX2_SRC"],
                     qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"], dataset_stats_path=os.environ["DATASET_STATS_PATH"])
frame = np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8)
obs = {"view1": torch.from_numpy(center_crop_resize(frame, 224, 224)),
       "view2": torch.from_numpy(center_crop_resize(frame, 224, 224)), "proprio": np.zeros(8, dtype=np.float32)}
pre = "A video recorded from a robot's point of view executing the following instruction: "
goals = ["turn on the stove and put the moka pot on it",
         "Place the black bowl into the open bottom drawer of the grey cabinet and close that drawer",
         "put the red-and-white alphabet soup carton in the woven basket with the silver rim",
         "put the red tomato sauce bottle in the woven basket with the silver rim"]
def timed(f):
    torch.cuda.synchronize(); t0 = time.perf_counter(); f(); torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3
def tick():
    fe.stage_inputs(obs); fe._graph.replay()
for rep in range(2):
    for g in goals:
        sp = timed(lambda: fe.set_prompt(pre + g))
        t1 = timed(tick)
        t2 = timed(tick)
        print(f"rep {rep} set_prompt {sp:.1f} ms, first tick {t1:.1f} ms, next tick {t2:.1f} ms | {g}", flush=True)
print(f"allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB, reserved {torch.cuda.memory_reserved() / 1e9:.2f} GB")
EOF
```

带回：8 行 `set_prompt` / 第一次 tick / 下一次 tick 的 ms；`allocated` / `reserved`；`H2_free.log`。

判据：`set_prompt` 分三类读——进程内第一次、第一次出现的指令长度、见过的长度或重复的指令，与 5090 的 751 / 240 / 30 ms 对照。第一次 tick 明显长于下一次 tick 时，说明换长度后的 graph 捕获推迟到了第一次推理，两者要相加。`reserved` 加 27B 规划器约 28 GB（NVFP4 权重，另加 KV），和 `free` 的可用量比较，判断规划器、执行器、编码器三者能否同时常驻。

结论去向：`opportunities.md` OPT-040（执行器的指令通路）；`research/reports/planner_executor_colocation_5090.md` 第 5 章痛点 5 的 Thor 部分；`THOR_STATUS_SUMMARY.md` 本轮小节。

### H1：按 harness 的节奏共置（执行器 533 ms 一个动作块，规划器每 5.33 s 一次）

问题：VoLoAgent harness 的执行器每 533 ms 要一个动作块（8 步 × 15 Hz），monitor 每 80 步（5.33 s）调用一次规划器。5090 上按这个节奏回放，prio 方式没有一次执行器查询超过 533 ms。Thor 上：执行器单独 fp8、10 步约 163 ms，余量约 3 倍，但 `1028` / `1029` 轮里长 prefill 期间有 1.2–2.4 s 的停顿（ISSUE-098），它会不会打破动作块的截止期？规划器替身用 Qwen3-VL-2B（每次 32 token），上下文约 4 k token，与 monitor 调用长度相同。

```
export OUT=$HOME/thor_val/$(date +%m%d)_h; mkdir -p $OUT/cells
cd $FLASHRT_ROOT/research/experiments/colocation_5090
export DROP_TEXT_ENCODER=1 NO_SMI=1 PERIOD_MS=533 PLANNER_CTX_EXTRA=3800
export DATASET_STATS_PATH=${DATASET_STATS_PATH:-$(dirname $CKPT_PATH)/dataset_stats.json}
run() { python colocation_bench.py "$1" "$2" 120 $OUT/cells/H1_$3.json > $OUT/cells/H1_$3.log 2>&1; grep -E "hz=|Error|Segmentation" $OUT/cells/H1_$3.log; }
tegrastats --interval 100 --logfile $OUT/H1_tegrastats.log & TS=$!
run isolated 0 isolated_start
run prio 0.1875 prio
run green25 0.1875 green25
export CUDA_MPS_PIPE_DIRECTORY=$OUT/.mps/pipe CUDA_MPS_LOG_DIRECTORY=$OUT/.mps/log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY; nvidia-cuda-mps-control -d
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50 run mps50 0.1875 mps50
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 run mps100 0.1875 mps100
echo quit | nvidia-cuda-mps-control; sleep 3; unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
run isolated 0 isolated_end
kill $TS
python - $OUT/cells/H1_*.json <<'EOF' | tee $OUT/H1_summary.txt
import json, sys
import numpy as np
for f in sys.argv[1:]:
    d = json.load(open(f)); lat = np.array(d["latency_series_ms"]); pc = d["planner_records"]
    over = [t for t, x in zip(d["ticks_wall_s"], lat) if x > 533]
    in_pf = sum(any(c["t_start"] <= t[0] <= c["t_start"] + c["prefill_ms"] / 1e3 for c in pc) for t in over)
    call = [(c["t_end"] - c["t_start"]) * 1e3 for c in pc]
    print(f"{f.rsplit('/', 1)[1]}: ticks {len(lat)}, p50 {np.percentile(lat, 50):.1f}, p99 {np.percentile(lat, 99):.1f}, "
          f"max {lat.max():.1f} ms, >533 ms {len(over)} (in planner prefill {in_pf}) | planner calls {len(pc)} "
          f"({pc[0]['prompt_tokens'] if pc else 0} prompt tokens), "
          f"call p50 {np.median(call) if call else 0:.0f} / max {max(call) if call else 0:.0f} ms, "
          f"prefill p50 {d['planner_prefill_ms_p50']}, decode p50 {d['planner_decode_ms_p50']}")
EOF
```

带回：`H1_summary.txt` 全部 6 行；每格日志里的 `hz=` 行；`H1_tegrastats.log` 中 VDD_GPU 与 RAM 的整段平均。

判据：
- 一格里没有任何查询超过 533 ms，才算守住动作块截止期；与 5090 回放（prio：0 次超过 533 ms）对照。
- 超过 533 ms 的查询如果落在规划器 prefill 区间内（`in planner prefill`），说明 ISSUE-098 的停顿就是 Thor 上打破 harness 节奏的原因。
- p99 相对 `isolated` 的倍数：只有比 `isolated_start` 与 `isolated_end` 之间的漂移更大的差别才能读（`1029` 轮漂移约 ±20 %）。
- 规划器单次调用 p50 与 5 333 ms 比较：超过它，说明即使是 2B 的替身，也跟不上每 80 步一次的 monitor（决策延迟 = 调用时间）。

结论去向：`THOR_STATUS_SUMMARY.md` 本轮小节；`research/experiments/colocation_5090/README.md` 的 Thor 一节；`research/reports/planner_executor_colocation_5090.md` 第 8 章；出现停顿时补进 issues.md ISSUE-098。

### H3（等提交推送后再做）：27B 规划器在 Thor 上按 harness 调用形状的单次耗时

问题：Qwen3.8-27B NVFP4 规划器在 Thor 上一次 decompose（约 2.1 k token、解码 24 个）和 monitor（约 4.1 k / 5.7 k token、解码 52–87 个）要多久，能否在 5.33 s 的 monitor 间隔内返回。上游文档里 Qwen3.6-27B（同一架构）在 Thor 上 2 k / 8 k 的 prefill 分别为 3.46 / 9.78 s（重复文本提示，`docs/qwen36_nvfp4.md`）。

前置：(1) 27B 规划器、图像前端、harness 回放这些提交已推送；(2) Thor 上有转换好的 NVFP4 权重（从公开的 `Qwen/Qwen3.8-27B-FP8` 下载约 31 GB，用 `research/experiments/qwen38_distill_bench/fp8_to_nvfp4_ct.py` 转换，输出 27.7 GB，转换时常驻内存约 30 GB），先确认磁盘与内存余量；(3) 图像前端目前只有 RTX 版，按调用形状回放的 Thor 脚本在推送时一并提供（5090 导出每个调用的图像特征与 RoPE 行，Thor 上用 `Qwen36TorchFrontendThor` 注入，ViT 时间另计）。命令与判据随该提交补到本项。

## 已完成的轮次（不再重跑）

逐轮结论已按"用法"第 2、3 条落库，不在本清单重复：每轮做了什么、数字是多少、口径是什么，看 `THOR_STATUS_SUMMARY.md` 的同名轮次小节（`eccf14f`、`a84916a`／`0919e`、`0920`、`0920s4`、`0920t`、`0920c`、`0921`、`0921x`、`0922`、`0922b`、`0922d`、`0922e`、`0922f`、`0922g`、`0922h`、`0922i`、`1021`、`1028`、`1029`、`1029b`、`1030`），各项结论看 `opportunities.md` 对应 OPT 条目。逐字的原始记录看 `git log`；本清单只保留"还没做"的东西。

---

## 已决定、不在这里测的

`text_trim` 转默认（原 E1）已经决定并落库：`default` profile 带 `text_trim=True`，新增 `native` profile；门禁与矩阵的默认口径跟着服务默认走（FA4 mot 与原生 VAE 进图后来也提升进了 `default`，`0921` 已在 Thor 上观察）；这些结论在 `plan.md` 的 "Decisions pending"、`opportunities.md` OPT-019/030 与 `THOR_STATUS_SUMMARY.md`。

三项 owner 决定（默认提升为最快配置、延迟基线按配置、标定 identity 加入相机几何）已落库，见 `plan.md` 的 "Decisions pending" 与 F1–F3；它们的 Thor 观察已在 `0921` 一轮完成。`plan.md` 的 "Open" 现在只剩目标工作负载自身未确认的声明（指令 token 数是否就是 128、checkpoint / 标定文件 / 图显存预算），它不是 Thor 测试项。

"Plan: ActionDiT small-M CUTLASS tile selection" 的 Thor 相位已在 `1021` 轮完成（X16）；`gemm_variant_autotune` 已按用户决定设为 nvfp4 的服务默认（`plan.md` "Decisions pending"）。
