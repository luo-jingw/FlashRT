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

行有效性：`rc=0`，`FA4 bb` / `FA4 mot` 与该行的定义一致，`FA4 fallback` 为 `None`。不满足的行作废重跑。

判据（阈值是工作值，按跑间波动调整）：
- 每行报告边际 = 本行 − 上一行，并同时报告 `stack` 相对 `default` 的实际差与各边际之和（差距就是重叠部分）。
- FA4 转默认的条件：`stack` 相对 `vae_trim` 的 P50 至少低 2 ms（约等于跑间波动），且 vs official 不劣于 `vae_trim`。否则 FA4 保持关。
- 原生 VAE 转默认的条件：`vae` 相对 `default` 的 P50 降低，且 vs official 不劣于 `default`。
- 精度行：各行 vs official 不低于 `tests/fixtures/imagewam_gate/fidelity_thresholds.json` 的阈值。
- 新入口：记录值的复现见 C1，`effective_config` 与解析器的一致见 C2，runtime identity 见 C3。

去向：OPT 条目写数字，`plan.md` 的 "Decisions" 写默认与 profile 的决定。

### C1 新入口下复现记录值（W12）
`default` 与 `stack` 是记录值 203.3 ms、106.1 ms 的来源（nvfp4，libero_spatial，`THOR_STATUS_SUMMARY.md`）；`PROFILES="default fast"` 的两行是同一组开关（`fast` 等于 `stack`），因此两者复现同一对记录值。本节第一条完整阶梯命令已包含 `default` 与 `stack`，可直接用它生成的表；只补这两行时：

```
SUITE=libero_spatial PRECS=nvfp4 ROWS="default stack" bash scripts/imagewam_thor_matrix.sh
```

判据：`default` 与 `stack` 的 P50 与 203.3 ms、106.1 ms 的差都在 2 ms 的跑间波动内（与 FA4 转默认判据同一工作值）；两行 `vs official` 的中位数不低于记录值 0.9976 与 0.99933；行有效性同本节。
去向：复现则写 `plan.md` 的 W12 相位状态；不达标则写 `issues.md`。

### C2 `effective_config` 与解析器逐字符一致（W12）
新入口下 profile 是部署记录的一部分：比较脚本打印的 `effective_config` 行必须与 `config_resolver.format_effective_config` 为同一 resolved 配置产出的字符串完全相同。解析器不知道运行时才定的 `use_fa4`（`default` profile 打印 `auto`）与 `fa4_fallback_reason`，比较时把 frontend 的 `fe.use_fa4` / `fe.use_fa4_mot` / `fe.fa4_fallback_reason` 传进去。环境变量用 C 节 profile 行那一轮的环境（`CKPT_PATH`、`FLUX2_*`、`QWEN3_MODEL_SPEC`、`PYTHONPATH`、`BUNDLE`、`OUT`）。

```
grep '^effective_config' $OUT/C_profile/matrix_libero_spatial_nvfp4_profile_default.log | tail -1 > $OUT/C2_default_script.txt
PROFILE=default python - > $OUT/C2_default_resolver.txt <<'PY'
import os
from flash_rt.frontends.torch.imagewam_thor import load_imagewam
from flash_rt.models.imagewam.config_resolver import format_effective_config, resolve_config
from flash_rt.models.imagewam.structure import ImageWAMStructure
from flash_rt.models.imagewam.workload import ImageWAMWorkload

w = ImageWAMWorkload.libero()
st = ImageWAMStructure.from_checkpoint(os.environ["CKPT_PATH"])
opts = dict(profile=os.environ.get("PROFILE", "default"), precision="nvfp4",
            ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"])
r = resolve_config(w, st, **opts)
fe = load_imagewam(os.environ["CKPT_PATH"], w, structure=st, flux2_src=os.environ["FLUX2_SRC"],
                   qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"],
                   dataset_stats_path=os.path.join(os.path.dirname(os.environ["CKPT_PATH"]),
                                                   "dataset_stats.json"),
                   **opts)
print(format_effective_config(r.options, use_fa4=fe.use_fa4, use_fa4_mot=fe.use_fa4_mot,
                              fa4_fallback_reason=fe.fa4_fallback_reason))
PY
diff -u $OUT/C2_default_resolver.txt $OUT/C2_default_script.txt
```

判据：`diff` 退出码 0；`fast` 行同样比较（`PROFILE=fast`，日志换成 `matrix_libero_spatial_nvfp4_profile_fast.log`，输出文件换名）。
去向：一致则并入 W12 的相位状态；不一致则写 `issues.md`，`diff` 打出的差异就是缺口。

### C3 runtime identity 带 `workload.<field>`（W10）
`from_config` 或 `load_imagewam` 构建的 frontend 在导出 runtime / ABI 描述时，`setup_identity` 在原有 `dims.<key>` 之外带九个 workload 字段：`workload.num_views`、`workload.image_h`、`workload.image_w`、`workload.text_max_len`、`workload.action_horizon`、`workload.action_dim`、`workload.proprio_dim`、`workload.num_steps`、`workload.shift`，值与 `ImageWAMWorkload.libero()` 的同名字段一致。这些条目是附加描述，不改 `calibration_file.IDENTITY_DIM_KEYS`，已记录的校准文件仍然有效。

```
python tests/gate_imagewam_model_runtime_export.py --precision nvfp4 2>&1 | tee $OUT/C3_export.log
python tests/gate_imagewam_native_schema_parity.py --precision nvfp4 2>&1 | tee $OUT/C3_native_schema.log
python - > $OUT/C3_export_identity.log 2>&1 <<'PY'
import os
from flash_rt.frontends.torch.imagewam_thor import load_imagewam
from flash_rt.models.imagewam.workload import ImageWAMWorkload

fe = load_imagewam(os.environ["CKPT_PATH"], ImageWAMWorkload.libero(), profile="default",
                   precision="nvfp4", ae_model_path=os.environ["FLUX2_AE_MODEL_PATH"],
                   flux2_src=os.environ["FLUX2_SRC"],
                   qwen3_model_spec=os.environ["QWEN3_MODEL_SPEC"],
                   dataset_stats_path=os.path.join(os.path.dirname(os.environ["CKPT_PATH"]),
                                                   "dataset_stats.json"))
fe.set_prompt("pick up the black bowl between the plate and the ramekin and place it on the plate")
print(fe.export_model_runtime().identity)
PY
grep -o 'workload\.[a-z_]*' $OUT/C3_export_identity.log | sort | uniq -c
```

判据：`C3_export.log`、`C3_native_schema.log` 都以 PASS 结束；`C3_export_identity.log` 里九项 `workload.*` 齐全且值等于 `ImageWAMWorkload.libero()` 的同名字段，`dims.*` 条目仍在。
去向：`plan.md` 的 W10 与 W12 相位状态；缺项或值不对写 `issues.md`。

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

说明：目标配置按 `ImageWAMWorkload` 的字段填入：部署方给出 `num_views`、每视角 `image_h`/`image_w`、`text_max_len`、`action_horizon`、`action_dim`、`proprio_dim`、`num_steps`、`shift`，`x0`、`img_len`、`a0`、`total`、`ref_h`、`ref_w`、`dt` 与 `vae_graph_input` 由它派生并在不一致时报错，不再手改 dims。

表内还缺三个 workload 字段，要由目标配置的所有者给定：`action_dim`（反归一化后的动作维度）、`shift`（噪声调度位移）、`text_max_len`（填充后的文本长度，与表内的指令 token 数 min / median / max 不是同一个量）；分辨率一栏写成每个视角的 `image_h`×`image_w`。`num_train_timesteps` 用默认 1000，除非目标 checkpoint 另有记录。

说明：`benchmarks/imagewam_e2e_official_compare.py`（矩阵脚本用的）读取 LIBERO 数据，dims 取自 `flash_rt/models/imagewam/libero_dims.py` 的 `LIBERO_REAL_DIMS`，即 LIBERO 工作负载。目标配置的延迟部分可以用 `benchmarks/imagewam_thor_graph_bench.py`（随机权重，dims 换成目标工作负载，不含 VAE 与 proprio，也无法测 `text_trim`）；精度与 trim 需要目标配置的数据和官方对照，这个缺口要在配置表填完之后再补。

判据：`ImageWAMWorkload` 的九个字段都能从表里取到值，且 `ImageWAMWorkload(...)` 加 `resolve_config(..., profile=...)` 不抛 `ConfigError`（`layout()` 与 `vae_graph_input()` 也不报错）。
去向：`plan.md` 的 W12 相位状态。

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
规则 R5 同样在 `consumer="abi"` 与 `consumer="native"` 下拒绝 `text_trim=True`，所以含 `text_trim` 的 profile（`fast`）不能用于 ABI 与 native 两条服务路径；需要这两条路径的部署在条件 5 落地前不能设 `text_trim`。

### E2 重定 Thor 延迟基线
矩阵结果稳定后，更新 `tests/fixtures/imagewam_gate/latency_baselines.json` 的 Thor 项（当前 nvfp4 门限 243 ms 对应旧基线 231.6 ms）。
