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

环境变量与构建见 `scripts/imagewam_thor_validation.sh` 文件头。本轮改动是 Python 与脚本，不含 kernel 改动，不需要重编。

```
export OUT=$HOME/thor_val/$(date +%m%d)
python -c "from flash_rt.hardware.jetson_clock_state import report_jetson_clock_state as r; r()" | tee $OUT/P0_clock.log
git rev-parse HEAD | tee $OUT/P0_commit.log
(cd $BUNDLE && sha256sum -c SHA256SUMS) | tee $OUT/P0_bundle.log
```

记录：commit hash；GPU 是否被其他进程占用；`emc_locked` 是否为 `null`（目前未锁定）。这三项写在本轮所有数字旁边。

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

固定负载，一行一个配置，表格由脚本从日志生成（`$OUT/matrix_<suite>_<tag>.md/.csv`）。行定义见 `scripts/imagewam_thor_matrix.sh` 文件头。本轮起矩阵的每一行都经新入口构建 frontend：`benchmarks/imagewam_e2e_official_compare.py` 用 `load_imagewam(ckpt_path, workload, profile=..., precision=..., calibration_path=...)` 构建，环境变量 `PROFILE` 选具名 profile（未设时为 `default`），原有的逐开关环境变量（`TEXT_TRIM`、`FA4_MOT`、`VAE_GRAPH`、`VAE_ENCODER`、`VAE_RESIZE`、`PRECISION`、`CALIBRATION`、`NVFP4_AWQ`）作为 expert 覆盖传入。

```
# 完整阶梯 + leave-one-out（libero_spatial，nvfp4）
SUITE=libero_spatial PRECS=nvfp4 bash scripts/imagewam_thor_matrix.sh
# trim 的收益随指令长度变化，另外两套只测三行
SUITE=libero_goal PRECS=nvfp4 ROWS="default vae_trim stack" bash scripts/imagewam_thor_matrix.sh
SUITE=libero_10   PRECS=nvfp4 ROWS="default vae_trim stack" bash scripts/imagewam_thor_matrix.sh
# 精度行：默认与叠满两档
SUITE=libero_spatial PRECS="e0m3_hadamard fp8_static_cutlass fp16" ROWS="default stack" bash scripts/imagewam_thor_matrix.sh
# profile 行：一行是一个具名 profile，不是一组开关；单独输出目录，避免与上面的开关行写出同名日志
OUT=$OUT/C_profile SUITE=libero_spatial PRECS=nvfp4 PROFILES="default fast" bash scripts/imagewam_thor_matrix.sh
OUT=$OUT/C_profile SUITE=libero_goal   PRECS=nvfp4 PROFILES="default fast" bash scripts/imagewam_thor_matrix.sh
OUT=$OUT/C_profile SUITE=libero_10     PRECS=nvfp4 PROFILES="default fast" bash scripts/imagewam_thor_matrix.sh
```

`PROFILES` 模式一行是一个具名 profile（`default`、`fast`），表与 CSV 里的行名是 `profile_<name>`，日志是 `matrix_<suite>_<prec>_profile_<name>.log`，与开关行的同名日志不冲突；表格文件名也带 mode 与 profile 名。不加 `PROFILES` 时仍按 `ROWS` 的开关行跑，行名与含义不变（`default vae vae_trim vae_trim_fa4bb stack stack_no_vae stack_no_trim`）。profile 行只导出 `PROFILE`，不导出开关变量，所以 `fast` 一行跑的是 profile 自己的内容（等于 `stack` 那一组），不是 profile 再叠脚本的默认值。

已完成（`c20f3a0`，数字在 `opportunities.md` OPT-019/021/030 与 `THOR_STATUS_SUMMARY.md`）：`libero_spatial` 的 nvfp4 完整阶梯与 leave-one-out，以及它的 `PROFILES="default fast"` 两行。仍需跑：`libero_goal`、`libero_10` 各三行，精度行，以及 `libero_goal` / `libero_10` 的 profile 行。

行有效性：`rc=0`，`FA4 bb` / `FA4 mot` 与该行的定义一致，`FA4 fallback` 为 `None`。不满足的行作废重跑。

判据（阈值是工作值，按跑间波动调整）：
- 每行报告边际 = 本行 − 上一行，并同时报告 `stack` 相对 `default` 的实际差与各边际之和（差距就是重叠部分）。
- FA4 转默认的条件：`stack` 相对 `vae_trim` 的 P50 至少低 2 ms（约等于跑间波动），且 vs official 不劣于 `vae_trim`。否则 FA4 保持关。2 ms 这个工作值要在 E2 的重复测量给出会话内极差之后重定：`c20f3a0` 一轮里同一组开关出现两次，相差 13.6 ms（ISSUE-082），比 2 ms 大。
- 原生 VAE 转默认的条件：`vae` 相对 `default` 的 P50 降低，且 vs official 不劣于 `default`。
- 精度行：各行 vs official 不低于 `tests/fixtures/imagewam_gate/fidelity_thresholds.json` 的阈值。
- 新入口：`c20f3a0` 一轮里已做过三项检查——记录值复现（`fast` 106.8 ms 落在 106.1 的波动内；`default` 的同口径基线缺失，见 ISSUE-082）、`effective_config` 与 `config_resolver` 逐字符一致（`default` 与 `fast`）、runtime identity 带九个 `workload.<field>` 且 ABI 与 `infer()` bit-exact、native schema 与 golden 一致。结论在 `plan.md` 的 W12、`opportunities.md` 的 OPT-019/021/030 与 `issues.md` ISSUE-082。

去向：OPT 条目写数字，`plan.md` 的 "Decisions" 写默认与 profile 的决定。

---

## D. 目标配置（非 LIBERO）

先填这张表，再决定怎么测：

| 参数 | 值 |
|---|---|
| 相机数 × 分辨率 | 3 × 256×256 |
| 指令 token 数（min / median / max） | 16 / 待定 / 128 |
| 文本缓冲长度 `text_max_len` | 128（待确认，见 ISSUE-083：live Qwen3 固定输出 512 行） |
| action horizon | 32 |
| 去噪步数 | 10 |
| proprio 维度 | 8：必须等于目标 checkpoint 的 `proprio_encoder` 宽度（LIBERO 微调是 7 关节 + 1 夹爪）。双臂 29 DoF 需要与它匹配的 checkpoint；模型其余部分与这一维无关，前端会用真实权重形状校验 |
| 延迟预算 | 不设：只记录数字，不做判别 |
| 服务路径 | 三条都测（Python `infer()` / ABI / native），作为三个配置对比 |
| checkpoint 与校准文件 | checkpoint 复用 FLUX.2-4B 的 `model.pt`；校准文件待目标数据 |
| 图显存预算 | 待定 |

派生（由 `ImageWAMWorkload` 计算，不手填）：`ref_h=16`、`ref_w=48`、`img_len=768`、`text_max_len=128` 时 `x0=129`、`a0=897`、`total=929`、`dt=0.1`。

说明：目标配置按 `ImageWAMWorkload` 的字段填入：部署方给出 `num_views`、每视角 `image_h`/`image_w`、`text_max_len`、`action_horizon`、`action_dim`、`proprio_dim`、`num_steps`、`shift`，`x0`、`img_len`、`a0`、`total`、`ref_h`、`ref_w`、`dt` 与 `vae_graph_input` 由它派生并在不一致时报错，不再手改 dims。`action_dim=7`、`shift=5.0` 随候选值给定。

跑之前先看三个已知缺口（都在 `issues.md`）：

- **ISSUE-084**：observation 与图外 VAE 编码通路目前只支持 1–2 路视角，三路 workload 走 `infer()` 且带真实 VAE 会报 `expected 3 views, got 2`；今天能测的是"三路序列布局 + 随机 image tokens"的延迟，测不到 VAE stage。
- **ISSUE-083**：`text_encoder.py` 固定 `max_length=512`，`text_max_len=128` 的 workload 需要先决定 context 怎么产生（截取前 128 行 / 改编码长度 / 保持 512 再 trim）。
- **ISSUE-081**：VAE 不在图内时 ABI 的 `view_shape` 硬编码 `(2,224,224)`，三路 256 的 ABI 端口形状不对。

精度与 trim 需要目标配置的数据和官方对照，这个缺口要在配置表确认之后再补；延迟部分用下节的 path bench。

判据：`ImageWAMWorkload` 的九个字段都能从表里取到值，且 `ImageWAMWorkload(...)` 加 `resolve_config(..., profile=...)` 不抛 `ConfigError`（`layout()` 与 `vae_graph_input()` 也不报错）。
去向：`plan.md` 的 W12 相位状态。

---

## P. 服务路径对比（同一 workload，三条路径）

`benchmarks/imagewam_thor_path_bench.py`：不设 `CKPT_PATH` 时用随机权重、随机帧、随机 context，因此只测延迟；每个 workload 一个表，`infer()` / ABI（`io="python"`）/ native（`io="native"`）各一行 P10/P50/P90 与 n，另打印 workload 字段、派生布局、`effective_config` 行与 Jetson 时钟状态。构不出来的路径会打出原因并跳过，不影响其余路径。

`--valid-tokens` 决定每个 prompt 的有效文本 token 数，也就是 trim 对比真正在比的东西：开 trim 时序列是 `valid+1` 行，关 trim 时永远是 `text_max_len+1` 行。给一个逗号列表就扫一遍（每换一个长度会触发一次捕获，打印里带 `set_prompt` 耗时）。默认值按 workload 取（LIBERO 24，目标 32）。

```
# LIBERO workload（2 × 224×224，text 512，horizon 64）
python benchmarks/imagewam_thor_path_bench.py                                   # profile default
python benchmarks/imagewam_thor_path_bench.py --profile fast                    # trim + FA4 + VAE 进图
python benchmarks/imagewam_thor_path_bench.py --text-trim on --valid-tokens 16,24,31
python benchmarks/imagewam_thor_path_bench.py --paths infer,abi                 # 没建 native 时

# 目标 workload（3 × 256×256，horizon 32，指令 16–128 token）
python benchmarks/imagewam_thor_path_bench.py --workload target --text-max-len 128
python benchmarks/imagewam_thor_path_bench.py --workload target --text-max-len 128 --text-trim on \
  --valid-tokens 16,72,128
python benchmarks/imagewam_thor_path_bench.py --workload target --text-max-len 512 --text-trim on
```

ABI 一行需要 `exec/` 构建，native 一行需要 `runtime/` 与 `flashrt_imagewam_native`（见 `scripts/imagewam_thor_validation.sh` 文件头）。**开 trim 的配置下 ABI 与 native 两行今天会被跳过**（规则 R5，ABI/native 还没有"按长度一张图"的支持，见 S2）；等 S2 落地后同样的命令会给出三条路径的数。

判据：每条的 rc；被跳过路径打出的原因；三条路径的 P50 记在案。**不设阈值、不判定好坏**——目标配置没有延迟预算。
另外读三行：目标 workload 的 `view_shape`（默认 profile 下 VAE 在图外，应为 `(3, 256, 256)` 而不是 `(2, 224, 224)`，ABI 的 `images` 端口与 `views` 名单随之变成三路，见 ISSUE-081 的 Resolution）；`--workload target --text-max-len 128` 时 context 若是 `(1, 128, 7680)` 就满足 `x0 == text_len + 1`（ISSUE-083）；LIBERO 两路的 VAE token 与改动前逐位相同（跑一次 `tests/test_imagewam_vae_stage.py` 与一次 e2e 即可）。
去向：ABI 与 native 的数字进 `opportunities.md` 的 OPT-028 / OPT-029，汇总进 `THOR_STATUS_SUMMARY.md`。

判据：每条的 rc；被跳过路径打出的原因；三条路径的 P50 记在案。**不设阈值、不判定好坏**——目标配置没有延迟预算。
另外读三行：目标 workload 的 `view_shape`（默认 profile 下 VAE 在图外，应为 `(3, 256, 256)` 而不是 `(2, 224, 224)`，ABI 的 `images` 端口与 `views` 名单随之变成三路，见 ISSUE-081 的 Resolution）；`--workload target --text-max-len 128` 时 context 若是 `(1, 128, 7680)` 就满足 `x0 == text_len + 1`（ISSUE-083）；LIBERO 两路的 VAE token 与改动前逐位相同（跑一次 `tests/test_imagewam_vae_stage.py` 与一次 e2e 即可）。
去向：ABI 与 native 的数字进 `opportunities.md` 的 OPT-028 / OPT-029，汇总进 `THOR_STATUS_SUMMARY.md`。

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
规则 R5 同样在 `consumer="abi"` 与 `consumer="native"` 下拒绝 `text_trim=True`。这条拒绝是"尚未实现"（条件 5），不是 ABI 或 native 的性质：ABI 与 native 都是本项目自己的代码，按长度的图可以支持，见 ISSUE-080 末尾记的两种形状——精确长度 + 有界缓存（不改 kernel，图集合随数据变化），或 16 token 分块 + 对块内 padding 加掩码（图集合固定，LIBERO 33 张、128 token 的 workload 9 张，代价是掩码与 kernel 工作）。

### E2 重定 Thor 延迟基线
矩阵结果稳定后，更新 `tests/fixtures/imagewam_gate/latency_baselines.json` 的 Thor 项（当前 nvfp4 门限 243 ms 对应旧基线 231.6 ms）。

`c20f3a0` 一轮暴露出两个口径问题（ISSUE-082），重定基线前先补这两项测量：

```
# 1) 默认行的同口径基线：本 commit 的 gate 数，与记录的门 203.3 ms 对比；
#    不一致时在同一会话里再跑一次 pre-change commit 1ff6034 的同一命令。
python tests/gate_imagewam_libero.py --precision nvfp4 --fixture-dir "$BUNDLE/imagewam_libero_gate_v1" \
  --output-dir $OUT/E2_gate_nvfp4 2>&1 | tee $OUT/E2_gate_nvfp4.log

# 2) 同一配置的会话内离散度：同一行重复三次（以及换一个会话再跑一次），
#    连同 P0_clock 的状态一起记录。
SUITE=libero_spatial PRECS=nvfp4 ROWS="default stack" bash scripts/imagewam_thor_matrix.sh
```

判据：得到 `default` 行的同口径（gate）数字，以及同一行三次重复的极差；用这个极差替换 C 节的 2 ms 工作阈值，并据此更新 `latency_baselines.json` 的 Thor 条目与 `THOR_STATUS_SUMMARY.md` 的默认行。
去向：`issues.md` ISSUE-082、`plan.md` 的 W12 相位状态、`tests/fixtures/imagewam_gate/latency_baselines.json`。
