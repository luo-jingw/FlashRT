# Thor 测试清单

本节次（配置整合 + `text_trim` 服务化）剩余的待测项。已完成的项已按"用法"第 2 条删除，结论在下面表格指定的文件里。

## 用法

1. `git pull` 后按顺序做。每项的命令、判据、结论去向都写在项内。
2. 做完一项：结论写进"结论去向"指定的文件，然后**从本清单删除这一项**。没有结论的中间产物不保留。
3. 每轮结束提交删减后的清单。历史看 `git log`，结论看下表的文件。

原始日志只留在 `$OUT`，不进 git；只有回归基线进 git（`tests/fixtures/imagewam_gate/*.json`，fixture 数据本身留在 `$BUNDLE`）。

| 结论类型 | 写到 |
|---|---|
| 测得的数字与结论 | `opportunities.md` 对应 OPT 条目（OPT-016 … OPT-031） |
| ABI / native 服务路径的数字 | `opportunities.md` OPT-028（ABI）、OPT-029（native） |
| 默认值 / profile 决定、相位状态 | `plan.md` 的 "Execution status" 与 "Decisions pending" |
| 缺陷、未解释的现象、`text_trim` 转默认的条件 | `issues.md`（ISSUE-080、ISSUE-085、ISSUE-086 等） |
| 对外总结 | `THOR_STATUS_SUMMARY.md` |

## 已完成（`eccf14f` 一轮，不再重跑）

LIBERO nvfp4 阶梯（spatial 三次重复、goal、libero_10）、叠满精度表、gate nvfp4 202.2 ms、A2 fp16（306.6 → 234.9 → 222.9 ms）、B2（AWQ 0.99970 / 真实 FP8 校准 0.99997）、E2 同口径基线与会话内极差（default 0.4 ms、stack 0.5 ms，`latency_baselines.json` 已按 202.2 ms 重定）、E3 fixture v2 与 trim gate（vs official 0.99931 / min 0.99898，P50 114.6 ms）、S1 三组 pytest（ABI multilength 10、guards 7、cache 24）、LIBERO 三路径（infer 202.3 / ABI 184.2 / native 183.8；`fast` 93.2 / 95.1 / 跳过）、目标 workload 的 `view_shape=(3,256,256)` 与 ABI/native 152.7 / 154.5 ms。数字在 `opportunities.md` 与 `THOR_STATUS_SUMMARY.md`。

---

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

## A. 先确认数据可信

### A1 三个测试级失败是真问题还是级联（ISSUE-085）
`eccf14f` 一轮与上一轮相同：AWQ 测试的 plain 路径 kernel 计数 `plain=0`；FA4 dispatch 的 `capture_sync` 模式仍是 mempool 行为；graph-recover 的 `torch.equal` 仍失败。服务路径不受影响（整轮矩阵 `FA4 fallback=None`），但三个测试在 Thor 上是红的。
```
python -m pytest tests/test_imagewam_*.py tests/test_jetson_clock_state.py -x -q -rs 2>&1 | tee $OUT/A1_first_failure.log
python -m pytest tests/test_imagewam_fa4_dispatch.py -x -q 2>&1 | tee $OUT/A1_fa4_dispatch.log
python -m pytest tests/test_imagewam_fa4_dispatch.py -k capture_sync tests/test_imagewam_frontend.py -q 2>&1 | tee $OUT/A1_cascade.log
python -m pytest tests/test_imagewam_awq.py -q 2>&1 | tee $OUT/A1_awq.log
python -m pytest tests/test_imagewam_graph_recover.py -q 2>&1 | tee $OUT/A1_graph_recover.log
TRIM_PRECISION=nvfp4 TRIM_FA4=on python -m pytest tests/test_imagewam_text_trim_graph_safety.py -q -s -k capture_failure 2>&1 | tee $OUT/A1_fa4_recover.log
```
假设（未证实）：`test_imagewam_fa4_dispatch.py` 的 `capture_sync` 模式在捕获中调用 `torch.cuda.synchronize()`，使捕获状态失效，其后的 module 级 fixture 构造失败，表现为大量 `ERROR at setup`。
判据：`A1_fa4_dispatch.log` 里该文件单独跑就失败 → 真问题；单独过、`A1_cascade.log` 里后续测试失败 → 级联，污染源是 `capture_sync`；`A1_awq.log` / `A1_graph_recover.log` 单独跑的结果决定这两项是独立缺陷还是同一个捕获状态问题的下游。
去向：`issues.md` ISSUE-085。

---

## B. 补缺失数据

### B1 `e0m3_hadamard` 的 gate（本轮可跑：阈值已加）
`fidelity_thresholds.json` 现在有 `e0m3_hadamard`（与 nvfp4 同界）与 `fp8_static_cutlass`（与 fp8_static 同界）两条；此前 gate 报 blocked 只是因为按名字查不到条目。
```
python tests/gate_imagewam_libero.py --precision e0m3_hadamard --fixture-dir "$BUNDLE/imagewam_libero_gate_v1" --output-dir $OUT/B1_gate_e0m3 2>&1 | tee $OUT/B1_gate.log
python tests/gate_imagewam_libero.py --precision fp8_static_cutlass --calibration "$CAL" --fixture-dir "$BUNDLE/imagewam_libero_gate_v1" --output-dir $OUT/B1_gate_fp8cutlass 2>&1 | tee $OUT/B1_gate_fp8cutlass.log
```
判据：两条都过（不再出现 "no fidelity thresholds for precision"），并记录各行的 vs official 中位数与 P50；`e0m3_hadamard` 的 vs official 中位数不低于 nvfp4 的门 0.99744。
去向：OPT-024、OPT-022。

### B3 `text_trim` 每长度图显存增量（ISSUE-080 条件 6 的缺口）
捕获与切换耗时已有（`eccf14f`：预捕获后 0.000–0.012 s，首次捕获 0.42–0.58 s，见 OPT-030）；还缺每个长度占多少显存。
```
python benchmarks/imagewam_text_trim_bench.py --precision nvfp4 --section all --use-fa4 off --iters 20 --rounds 5 2>&1 | tee $OUT/B3_trim_bench.log
```
判据：读出每个缓存长度带来的进程显存增量（NVML/`torch.cuda.max_memory_allocated()` 前后差），与 `text_trim_cache_size` 的默认 32 相乘，写进 `issues.md` 条件 6 的 Resolution。
去向：ISSUE-080、OPT-030。

---

## P. 目标 workload 的三条路径（ISSUE-086 修完后）

LIBERO 两个 profile 与目标 workload 的 ABI/native 行已完成；目标 workload 的 `infer()` 因 VAE 几何（768 vs 588，ISSUE-086）被跳过，`profile=fast` 构造时报 `img_raw` 期望 `(588,128)` 得到 `(768,128)`。VAE 几何按 workload 的每视角尺寸修好后重跑：

```
# 目标 workload（3 × 256×256，horizon 32，指令 16–128 token）
python benchmarks/imagewam_thor_path_bench.py --workload target --text-max-len 128
python benchmarks/imagewam_thor_path_bench.py --workload target --text-max-len 128 --profile fast --precapture
python benchmarks/imagewam_thor_path_bench.py --workload target --text-max-len 128 --text-trim on \
  --valid-tokens 16,72,128 --precapture
# LIBERO：确认两路 VAE token 与改动前逐位相同
python -m pytest tests/test_imagewam_vae_stage.py -q 2>&1 | tee $OUT/P_vae_stage.log
python benchmarks/imagewam_thor_path_bench.py
```

判据：目标 workload 三条路径都出数（不再 SKIP，`view_shape=(3,256,256)`，`img_len=768`）；`--profile fast` 构造成功；LIBERO 行的 VAE token 与 `eccf14f` 的逐位相同（跑一次 e2e 对比 gate fixture 的 fp16 参考即可）。**不设阈值、不判定**——目标配置没有延迟预算。
去向：目标行进 `THOR_STATUS_SUMMARY.md`（三路数字）与 OPT-028/029/030；LIBERO 位一致性不达标则写 `issues.md`（ISSUE-086）。

---

## S. S 相位验证

### S1 驱逐路径（S3 的有界缓存）
上一轮的测法与实现冲突：`precapture_text_lengths` 声明的长度数超过 `text_trim_cache_size` 时构造期直接拒绝（否则填充循环会把刚驱逐的再捕回来，永不收敛），所以"预捕获 3 个长度 + 上限 2"测不到驱逐。改为**不预捕获**、用 `set_prompt` 逐长度首次捕获，让上限先被撑满：
```
python benchmarks/imagewam_thor_path_bench.py --text-trim on --valid-tokens 16,24,31 --text-trim-cache-size 2
```
三个长度总共只需两张图，第三个长度捕获时最久未用的那张被驱逐；再回到第一个长度时会重新捕获。
判据：日志里三个长度都有数；第三个长度首次 `set_prompt` 的耗时是捕获量级（0.4 s 上下），之后再回到第一个长度仍要重新捕获（耗时同量级，这是设计行为）；进程不因上限报错；若要看"预捕获后只切图"，另跑一条 `--precapture --text-trim-cache-size 8`（长度数 ≤ 上限）。
去向：`issues.md` ISSUE-080 条件 6、`plan.md` S3 相位状态。

---

## E. 收尾

### E1 `text_trim` 转默认的剩余条件（ISSUE-080）
六条里已完成 1、3、4、5 的 ABI 一半、6；未完成：
- 条件 2 的 FA4 开分支——依赖 A1/ISSUE-085 的结论。
- 条件 5 的另一半 native——C++ `NativeRuntime` 单图/单 `context_rows`，见 `plan.md` 的 S4 相位；在此之前 `consumer="native"` 仍被规则 R5 拒绝。

在条件 2 与 5 的 native 一半落地前，`text_trim` 不写成 native 路径的默认；Python `infer()` 与 ABI 两条路径已经可以。

### E2 fixture v2 的 manifest 进 git
数据已生成、gate 已过（E3 判据满足），只差把 manifest 提交并放回 bundle（在 Thor 侧执行）：
```
git add tests/fixtures/imagewam_gate/imagewam_libero_gate_v2.manifest.json
git commit -m "gate: fixture v2 manifest (trimmed fp16 reference)"
git push
# 数据目录与新的 SHA256SUMS 留在 $BUNDLE/imagewam_libero_gate_v2/
(cd $BUNDLE && sha256sum -k SHA256SUMS > /dev/null && echo bundle-ok)
```
判据：manifest 进 git 后 `tests/test_imagewam_regression_gate.py::test_committed_fixture_manifests_are_well_formed` 在 CI/本机都过（它遍历 `*.manifest.json`）。
去向：`issues.md` ISSUE-080 条件 4 的 Resolution 补一句"manifest 已入库"。
