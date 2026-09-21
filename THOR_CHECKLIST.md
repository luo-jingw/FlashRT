# Thor 测试清单

0921 一轮的待测项已做完并落库（`THOR_STATUS_SUMMARY.md` 的 `0921` 小节）。本节次有两组：**L0–L6：最终结果表的 LIBERO 一张**（官方 torch 对 FlashRT fp16 / fp8 / fp4 / int8 / int4，稳态延迟，10 步；表的格式与规则见 `benchmarks/imagewam_result_table.py`，方案见 `plan.md` 的 "Plan: the final result tables"），以及 **P1：ISSUE-087 的探针**。RoboTwin 那一张（30 步）要等它的 workload 声明，不在本清单。

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

### L0–L6 最终结果表，LIBERO（10 步）：一个 session，按顺序连续跑

原则：**同一个 session 内一口气跑完**（表里只有同一 session 的行才会算相对官方的倍数）；期间不要跑别的 GPU 任务；日志名固定。session 的口径：GPU 空闲、`emc_locked` 记录、MAXN、GPU 锁频（`P0_clock.log`）。计时边界对所有行相同：相机帧 + proprio（文本已编码）→ 反归一化 action，稳态；不含 Qwen3。

```
export SESSION=$(date +%m%d)_final
export OUT=$HOME/thor_val/$SESSION; mkdir -p $OUT
CAL_TRIM=$BUNDLE/imagewam_libero_calib_n64_trim_v2.safetensors
```

**L1 官方 torch（baseline 行）**
```
python benchmarks/imagewam_official_torch_bench.py --workload libero --warmup 5 --iters 30 2>&1 | tee $OUT/L1_official.log
```
环境变量同 `imagewam_e2e_official_compare.py`（`CKPT_PATH`、`FLUX2_SRC`、`FLUX2_MODEL_PATH`、`FLUX2_AE_MODEL_PATH`、`QWEN3_MODEL_SPEC`）。要的数：最后一行的 `P10 / P50 / P90 / n`。参照：之前另一次会话测的官方 P50 是 453.6 ms；差得多要在报告里说。

**L2 FlashRT fp16 / fp8 / fp4 的延迟**（服务 `default` profile：裁文本、FA4 双位点、原生 VAE 进图；`--precapture` 让首次捕获不进计时）
```
for P in fp16 nvfp4; do
  python benchmarks/imagewam_thor_path_bench.py --workload libero --profile default --precision $P --paths infer --precapture --warmup 20 --bench-iters 100 2>&1 | tee $OUT/L2_$P.log
done
python benchmarks/imagewam_thor_path_bench.py --workload libero --profile default --precision fp8_static_cutlass --calibration $CAL_TRIM --paths infer --precapture --warmup 20 --bench-iters 100 2>&1 | tee $OUT/L2_fp8.log
```
要的数：每行 `infer` 的 `P10 / P50 / P90 / n`，以及打印的 `effective_config`（必须是 `text_trim=True vae_encoder=native vae_graph=True use_fa4=True use_fa4_mot=True fa4_fallback_reason=None`；有回退这一行作废）。判据：`fp16` 应明显低于 fp16 未裁剪的 275 ms（裁剪与精度无关，之前 fp16 叠满 273.8 ms 的异常就是这个判据没满足）；不满足先停，把 `effective_config` 和日志带回，不要继续。

**L3 三个精度的精度指标**（真实 LIBERO 数据；`profile default`，同一 session；这是 `imagewam_e2e_official_compare.py` 的口径：vs official 的余弦与 MAE，它自己的延迟列不用于表）
```
SUITE=libero_spatial PRECS="fp16 nvfp4" PROFILES="default" TAG=L3 bash scripts/imagewam_thor_matrix.sh
CALIBRATION=$CAL_TRIM SUITE=libero_spatial PRECS="fp8_static_cutlass" PROFILES="default" TAG=L3_fp8 bash scripts/imagewam_thor_matrix.sh
```
要的数：`matrix_libero_spatial_*_L3*.md` 里每行的 vs official min / median、MAE vs GT median、样本数（`N_TASKS x FRAMES x SEEDS`）。判据：`rc=0`，`FA4 fallback` 为 `None`。

**L4 int8 / int4**（SM80 CUTLASS 的 GEMM 基准，随机 packed 操作数、无激活量化，是上界，不是全流程数字；两个各自一个进程）
```
python benchmarks/imagewam_thor_int8_bench.py 2>&1 | tee $OUT/L4_int8.log
python benchmarks/imagewam_thor_int4_bench.py 2>&1 | tee $OUT/L4_int4.log
```
要的数：每个日志里的 `prefill (VAE+txt_in+25L backbone): P50=...` 与 `one denoise step (25L ActionDiT): P50=...`（表里这一行 = prefill + 10 × 单步）。如果报缺 kernel，说明扩展没带 `ENABLE_SM80_INT8_CUTLASS` 构建，把报错带回，不要静默重编。

**L5 收尾复测（夹住 session 内漂移）**：把 L1 与 fp4 的 L2 各再跑一次，日志名 `L5_official.log`、`L5_nvfp4.log`。判据：两次的 P50 与开头的差在 1–2% 内，才认为同一 session 内可比；差更大就在报告里写出来。

**L6 带回什么**：`SESSION`、`P0_clock.log`、`P0_commit.log`、GPU 是否独占；L1/L2/L5 的 P10/P50/P90/n；L2 三行的 `effective_config`；L3 的表；L4 的两个 P50；官方与 FlashRT 用的 checkpoint 的 sha256 前 16 位（`sha256sum $CKPT_PATH | cut -c1-16`）。由有凭据的一方录进 `docs/imagewam_results.json`（先 `check` 再 `render`）。

### P1 ISSUE-087：失败的捕获是否让 CUDA generator 遗留"正在捕获"标志
背景：`0921` 的全量 pytest（torch 2.9.1）在 FA4 测试之后，`test_imagewam_infer_action_noise.py` 构造 frontend 时 `torch.randn` 抛 `Offset increment outside graph capture encountered unexpectedly`。开发机（torch 2.14）上三种失败捕获的方式都不会留下这个状态，所以只能在 Thor 上查。
```
python scripts/probe_capture_generator_state.py 2>&1 | tee $OUT/P1_probe.log
python -m pytest tests/test_imagewam_fa4_dispatch.py -x -q 2>&1 | tee $OUT/P1_fa4_dispatch.log
python -m pytest tests/test_imagewam_fa4_dispatch.py -k capture_sync tests/test_imagewam_infer_action_noise.py -q 2>&1 | tee $OUT/P1_cascade.log
```
判据：探针每个 case 打印 `randn afterwards: ok` 还是那条错误、以及"一次成功捕获之后"是否恢复；`P1_cascade.log` 里第二个文件是否出错。全 ok → 污染另有来源（把 `01_pytest.log` 里第一处 ERROR 之前最近一个失败的测试带回）。有 case 留下状态 → 判断是测试收尾要复位（fixture 里做一次成功的小捕获），还是 frontend 的回退路径要复位（`_capture_graph_or_fall_back`）。
去向：`issues.md` ISSUE-087。

---

## 已完成的轮次（不再重跑）

逐轮结论已按"用法"第 2、3 条落库，不在本清单重复：每轮做了什么、数字是多少、口径是什么，看 `THOR_STATUS_SUMMARY.md` 的同名轮次小节（`eccf14f`、`a84916a`／`0919e`、`0920`、`0920s4`、`0920t`、`0920c`、`0921`），各项结论看 `opportunities.md` 对应 OPT 条目。逐字的原始记录看 `git log`；本清单只保留"还没做"的东西。

---

## 已决定、不在这里测的

`text_trim` 转默认（原 E1）已经决定并落库：`default` profile 带 `text_trim=True`，新增 `native` profile；门禁与矩阵的默认口径跟着服务默认走（FA4 mot 与原生 VAE 进图后来也提升进了 `default`，见 N 节）；这些结论在 `plan.md` 的 "Decisions pending"、`opportunities.md` OPT-019/030 与 `THOR_STATUS_SUMMARY.md`。

三项 owner 决定（默认提升为最快配置、延迟基线按配置、标定 identity 加入相机几何）已落库，见 `plan.md` 的 "Decisions pending" 与 F1–F3；它们的 Thor 观察已在 `0921` 一轮完成。`plan.md` 的 "Open" 现在只剩目标工作负载自身未确认的声明（指令 token 数是否就是 128、checkpoint / 标定文件 / 图显存预算），它不是 Thor 测试项。

另一份 plan（"Plan: ActionDiT small-M CUTLASS tile selection"，roadmap item 1）还有一个面向 Thor 的相位是 `blocked`：`gemm_variant_autotune` 的 tile 扫描与 `infer()` A/B 从未在 Thor 上跑过（`benchmarks/imagewam_thor_small_m_tile_sweep.py`，`issues.md` ISSUE-023，`opportunities.md` OPT-018）。它不属于本清单的收尾范围（`fast`/`default` 都不开这个开关），列在这里只是为了不让人以为机器上的活已经全干完——如果要做，那是一个独立的短项。
