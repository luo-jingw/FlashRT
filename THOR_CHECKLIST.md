# Thor 测试清单

## 用法

1. `git pull` 后按顺序做。每项的命令、判据、结论去向都写在项内。
2. 做完一项：结论写进"结论去向"指定的文件，然后**从本清单删除这一项**。没有结论的中间产物不保留。
3. 每轮结束提交删减后的清单。历史看 `git log`，结论看下表的文件。

原始日志只留在 `$OUT`，不进 git；只有回归基线进 git（`tests/fixtures/imagewam_gate/*.json`）。

| 结论类型 | 写到 |
|---|---|
| 测得的数字与结论 | `opportunities.md` 对应 OPT 条目（OPT-016 … OPT-030） |
| 默认值 / profile 决定、路线图状态 | `plan.md` 的 "Execution status" 与 "Decisions" |
| 缺陷、未解释的现象、`text_trim` 转默认的条件 | `issues.md`（ISSUE-080 等） |
| 对外总结 | `THOR_STATUS_SUMMARY.md` |

## 前置（每轮一次）

环境变量与构建见 `scripts/imagewam_thor_validation.sh` 文件头。本轮只改了 Python 与脚本，不需要重编。

```
export OUT=$HOME/thor_val/$(date +%m%d)
python -c "from flash_rt.hardware.jetson_clock_state import report_jetson_clock_state as r; r()" | tee $OUT/P0_clock.log
git rev-parse HEAD | tee $OUT/P0_commit.log
(cd $BUNDLE && sha256sum -c SHA256SUMS) | tee $OUT/P0_bundle.log
```

记录：GPU 是否被其他进程占用；`emc_locked` 是否为 `null`（目前未锁定）。这两项写在本轮所有数字旁边。

---

## A. 先确认数据可信

### A1 pytest 失败是级联还是真问题
上一轮 `01_pytest.log` 只留下失败标题，没有第一处失败的报错文本，这次要把它留下来。本机（无 GPU）已排除一个原因：路由契约测试的 stub 缺 `_nvfp4_variant_index`，已修，且它在合并跑时不触发，解释不了 Thor 的 96 failed / 51 errors。
```
python -m pytest tests/test_imagewam_*.py tests/test_jetson_clock_state.py -x -q -rs 2>&1 | tee $OUT/A1_first_failure.log
python -m pytest tests/test_imagewam_fa4_dispatch.py -x -q 2>&1 | tee $OUT/A1_fa4_dispatch.log
python -m pytest tests/test_imagewam_fa4_dispatch.py -k capture_sync tests/test_imagewam_frontend.py -q 2>&1 | tee $OUT/A1_cascade.log
python -m pytest tests/test_imagewam_residual_norm_fusion.py -q 2>&1 | tee $OUT/A1_isolated.log
TRIM_PRECISION=nvfp4 TRIM_FA4=on python -m pytest tests/test_imagewam_text_trim_graph_safety.py -q -s -k capture_failure 2>&1 | tee $OUT/A1_fa4_recover.log
```
假设（未证实）：`test_imagewam_fa4_dispatch.py` 的 `capture_sync` 模式在捕获中调用 `torch.cuda.synchronize()`，使捕获状态失效，其后的 module 级 fixture 构造失败，表现为大量 `ERROR at setup`。
判据：`A1_fa4_dispatch.log` 里该文件单独跑就失败 → 真问题（第 292 行 `set_prompt("fallback")`）；单独过、`A1_cascade.log` 里后续测试失败 → 级联，污染源是 `capture_sync`；其余单独跑能过 → 只是文件顺序问题。
去向：`issues.md`。

### A2 fp16 叠满 273.8 ms 是否合理
```
SUITE=libero_spatial PRECS=fp16 ROWS="default vae_trim stack" bash scripts/imagewam_thor_matrix.sh
```
判据：trim 与精度无关，`vae_trim` 应明显低于 `default`（预期省 60 ms 以上）。不降则先核对 `effective_config` 里 `text_trim=True`，再跑 `python benchmarks/imagewam_text_trim_bench.py --precision fp16 --section all --use-fa4 off --iters 20 --rounds 5` 对比裁剪前后的 replay 耗时。
去向：`issues.md`（不合理时）或 OPT-030（合理时，写明原因）。

---

## B. 补缺失数据

### B1 `e0m3_hadamard` 的 Thor 精度
```
python tests/gate_imagewam_libero.py --precision e0m3_hadamard --fixture-dir "$BUNDLE/imagewam_libero_gate_v1" --output-dir $OUT/B1_gate_e0m3 2>&1 | tee $OUT/B1_gate.log
```
判据：gate 全过，并且 vs official 的 median 不低于 nvfp4（0.99744）。当前只有延迟（198.9 vs nvfp4 202.3 ms），没有精度。
去向：OPT-024。

### B2 AWQ 与 FP8 校准的精度数字
```
STEPS="4" OUT=$OUT/B2 BUNDLE=$BUNDLE bash scripts/imagewam_thor_validation.sh
```
`SUMMARY.txt` 现在保留 `04_fid_*` 的 `min= median=`、`MAE ratio`、`all finite`、`peak GPU mem` 行（之前只留了延迟行）。
判据：AWQ 相对 nvfp4 的 vs fp16 cosine 有提升且延迟不增；`fp8_static*` 真实校准与占位校准的差距。
去向：OPT-023、OPT-022。

### B3 `text_trim` 在 Thor 上的代价（ISSUE-080 里标为"Thor 未测"）
```
python benchmarks/imagewam_text_trim_bench.py --precision nvfp4 --section all --use-fa4 off --iters 20 --rounds 5 2>&1 | tee $OUT/B3_trim_bench.log
```
读全文，记录三项：新长度首次 `set_prompt` 的捕获耗时、已缓存长度的切换耗时、每个长度的图显存增量。
去向：ISSUE-080、OPT-030。

---

## C. 配置矩阵（核心表）

固定负载，一行一个配置，表格由脚本从日志生成（`$OUT/matrix_<suite>_<tag>.md/.csv`）。行定义见 `scripts/imagewam_thor_matrix.sh` 文件头。

```
# 完整阶梯 + leave-one-out（libero_spatial，nvfp4）
SUITE=libero_spatial PRECS=nvfp4 bash scripts/imagewam_thor_matrix.sh
# trim 的收益随指令长度变化，另外两套只测三行
SUITE=libero_goal PRECS=nvfp4 ROWS="default vae_trim stack" bash scripts/imagewam_thor_matrix.sh
SUITE=libero_10   PRECS=nvfp4 ROWS="default vae_trim stack" bash scripts/imagewam_thor_matrix.sh
# 精度行：默认与叠满两档
SUITE=libero_spatial PRECS="e0m3_hadamard fp8_static_cutlass fp16" ROWS="default stack" bash scripts/imagewam_thor_matrix.sh
```

行有效性：`rc=0`，`FA4 bb` / `FA4 mot` 与该行的定义一致，`FA4 fallback` 为 `None`。不满足的行作废重跑。

判据（阈值是工作值，按跑间波动调整）：
- 每行报告边际 = 本行 − 上一行，并同时报告 `stack` 相对 `default` 的实际差与各边际之和（差距就是重叠部分）。
- FA4 转默认的条件：`stack` 相对 `vae_trim` 的 P50 至少低 2 ms（约等于跑间波动），且 vs official 不劣于 `vae_trim`。否则 FA4 保持关。
- 原生 VAE 转默认的条件：`vae` 相对 `default` 的 P50 降低，且 vs official 不劣于 `default`。
- 精度行：各行 vs official 不低于 `tests/fixtures/imagewam_gate/fidelity_thresholds.json` 的阈值。

去向：OPT 条目写数字，`plan.md` 的 "Decisions" 写默认与 profile 的决定。

---

## D. 目标配置（非 LIBERO）

先填这张表，再决定怎么测：

| 参数 | 值 |
|---|---|
| 相机数 × 分辨率 | |
| 指令 token 数（min / median / max） | |
| action horizon | |
| 去噪步数 | |
| proprio 维度 | |
| 延迟预算 | |
| 服务路径（Python `infer()` / ABI / native） | |
| checkpoint 与校准文件 | |
| 图显存预算 | |

说明：`benchmarks/imagewam_e2e_official_compare.py`（矩阵脚本用的）读取 LIBERO 数据，`REAL_DIMS` 固定为 LIBERO。目标配置的延迟部分可以用 `benchmarks/imagewam_thor_graph_bench.py`（随机权重，改 `REAL_DIMS`，不含 VAE 与 proprio，也无法测 `text_trim`）；精度与 trim 需要目标配置的数据和官方对照，这个缺口要在配置表填完之后再补。

---

## E. 收尾

### E1 `text_trim` 转默认的剩余条件（ISSUE-080）
Thor 上已满足：条件 1（三套件 trim ≥ 未 trim，P50 更低）、条件 3。
未满足：
- 条件 2 的 FA4 开分支（依赖 A1）。
- 条件 4：fixture v2（带 trim 的 fp16 参考，`benchmarks/imagewam_gate_fixture_generate.py`）——代码与数据工作。
- 条件 5：runtime surface / ABI / native 支持按长度的图（目前 `runtime_surface()`、`pipeline_resources()` 直接拒绝 `text_trim=True`）——代码工作。
- 条件 6：有界的按长度图缓存与启动时使用 `precapture_text_lengths`（该函数已存在，缺的是缓存上限与启动流程）——代码工作。

条件 4、5、6 不是 Thor 测试，完成之前 `text_trim` 不能转默认。

### E2 重定 Thor 延迟基线
矩阵结果稳定后，更新 `tests/fixtures/imagewam_gate/latency_baselines.json` 的 Thor 项（当前 nvfp4 门限 243 ms 对应旧基线 231.6 ms）。
