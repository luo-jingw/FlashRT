# Thor 测试清单

`0921_final` 那一轮的结果已记入 `THOR_STATUS_SUMMARY.md`：官方 torch 456.97 ms（L1），`fp16` 在服务默认下 274.70 ms、与未裁剪几乎相同（L2 的先决判据没过，L2 余下与 L3–L5 停了，现记为 ISSUE-088），ISSUE-087 的探针跑完（Thor 上 `synchronize()` 打断捕获会留下 generator 标志，一次成功小捕获可复位）。本节次待测：**D 组（先做）**：查 fp16 为什么不随裁剪变快，并用哨兵找出污染 generator 的测试；**L 组**：最终结果表 LIBERO 一张，D 组做完后**整组重跑**（表里只有同一 session 的行才算相对官方的倍数，上一轮停在 L2，不能接着用）。RoboTwin 那张要等它的 workload 声明，不在本清单。

## 指挥令（给 Thor 上执行清单的 agent）

目标：一次到位拿到 LIBERO 那张最终表的全部数字，并把 RoboTwin 所需的配置信息从 Thor 上查出来。不要在中间停下来等确认。

1. **顺序（不要并行，GPU 上同时只跑一件事）**：`git pull` → 前置 → D1 → D2 → D3 → L 组整组（L1…L6，一个 session）→ 下面的 R 组（RoboTwin 信息盘点，只读文件，不占 GPU）。
2. **遇到异常不停**：某一行失败或数字反常（包括 fp16 不随裁剪变快，那是已知的 ISSUE-088），把命令、退出码、报错原文写进报告，然后继续下一行。只有环境坏了（`git pull` 失败、checkpoint 不见了、GPU 被别的进程占用、构建缺失）才停，并说明缺什么。
3. **不要提交、不要推送**；不要为了让某一行通过而改代码或改判据。发现清单里的命令有错，照原样跑一遍、把报错带回，然后另外给出你认为正确的命令并标明它是你改的。
4. **同一 session 的口径**：L 组整组之间不要跑别的 GPU 任务；D 组的 profiler 不要与 L 组交叠。
5. **一次性汇报**：全部跑完后发一份报告，含：`P0_*` 前置记录；D1/D2 的表与判据结论（一句话）；D3 被点名的测试 id 与复位后的 failed/passed/errors；L1–L6 要求的全部数；R 组的发现。原始日志留在 `$OUT`。

### R 组：RoboTwin 信息盘点（只读，不占 GPU；在 L 组之后做）

要回答的是：RoboTwin 版 ImageWAM 的标准配置是什么、材料在 Thor 上的哪里。什么都别猜，找不到就明说找不到。
```
find / -iname '*robotwin*' -not -path '*/proc/*' 2>/dev/null | head -60
ls -la $HOME/thor_bundle $HOME/checkpoints 2>/dev/null
grep -ril robotwin "$(dirname "$FLUX2_SRC")" 2>/dev/null | head -30
```
报告：
- 有没有 RoboTwin 的 ImageWAM checkpoint（`model.pt`、`config.yaml`、`dataset_stats.json`），路径与 `sha256sum <model.pt> | cut -c1-16`。没有就写"没有"。
- 该 `config.yaml` 里这些段的原文：`model`（`variant`、`action_dit_config`、`proprio_dim`、schedulers）、`data`（相机键、图像尺寸、`context_len`、`qwen_context_len`、动作 horizon、动作维度）、以及任何写着推理步数或 shift 的字段（`eval_num_inference_steps`、`infer_shift` 等）。
- 数据：RoboTwin 数据集的位置、`meta/info.json` 里的 `features`（图像键与形状、`observation.state` 维度、`action` 维度）、fps；每个 episode 的指令文本几条示例（用来估计有效 token 数的范围）。
- 官方 RoboTwin 评测脚本怎么拼相机（几路、是否沿宽度拼接、送进 VAE 前的尺寸），文件路径与相关几行原文。
- 官方 RoboTwin 推理用多少去噪步（我们的标准是 30，确认是否一致）。

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

### D 组：fp16 在 `text_trim` 下不变快（ISSUE-088）与污染 generator 的测试（ISSUE-087）

**D1 逐 kernel 看哪部分不随行数缩小**（`benchmarks/imagewam_graph_kernel_profile.py`：同一进程按有效 token 数各建一个 frontend，对捕获的图做 `torch.profiler`，打印图 replay 耗时、图内 kernel 总耗时与占比、kernel 数、按 GEMM / 注意力 / 其他的分类耗时和耗时最高的 12 个 kernel；FA4 关，去掉一个变量；随机权重即可，设了 `CKPT_PATH` 就用真权重）
```
for P in fp16 nvfp4 fp16_cutlass; do
  python benchmarks/imagewam_graph_kernel_profile.py --precision $P --valid-tokens 24,512 --use-fa4 off 2>&1 | tee $OUT/D1_$P.log
done
```
要的数：每个精度在 24 与 512 有效 token 下的 `graph replay ... ms; kernels inside ... ms (..%)`、分类耗时、top kernel 表。判据（读表，不是过/不过）：
- 两个长度下总 replay 是否相差（fp16 预计接近，nvfp4 预计差约 90 ms）；
- fp16 里哪些 kernel 的每次 replay 耗时在 24 与 512 下相同：那就是不随行数缩小的部分，写出它的名字与耗时；
- `kernels inside` 占 replay 的比例：显著低于 100% 说明图里有 kernel 之间的空隙；
- `fp16_cutlass`（CUTLASS 的 fp16 档位）随裁剪缩小而 `fp16`（cuBLASLt）不缩小，就指向 cuBLASLt 的算法选择。
如果 profiler 在图 replay 里抓不到 kernel（输出 0 个 kernel），把报错带回，改用 `nsys profile --cuda-graph-trace=node`。

**D2 同进程的 replay A/B**（`imagewam_text_trim_bench.py` 的 `ab`：同一 frontend，20 有效 token 与 512 有效 token 交替，图 replay 用 CUDA events）
```
for P in fp16 nvfp4; do
  python benchmarks/imagewam_text_trim_bench.py --precision $P --section ab --use-fa4 off --iters 20 --rounds 5 2>&1 | tee $OUT/D2_$P.log
done
```
要的数：两边的 replay 时间。fp16 两边相同、nvfp4 差很多，就与上一轮一致，D1 的表给出原因；fp16 两边就不同，说明问题在 `path_bench` / `set_prompt` 的路径，而不在图。

**D3 污染 generator 的测试**（`tests/conftest.py` 的哨兵：每个测试后做一次 `normal_()`，第一个让 generator 停在"正在捕获"状态的测试在它的收尾报错并点名，同时用一次成功小捕获复位，后面的测试各自判定）
```
FLASHRT_GENERATOR_SENTINEL=1 python -m pytest tests/test_imagewam_*.py tests/test_jetson_clock_state.py -q -rfE 2>&1 | tee $OUT/D3_sentinel.log
```
要的数：日志里含 "left the CUDA generator in the capturing state" 的测试 id（可能不止一个）；总数 `failed / passed / errors`，与上一轮的 86 / 631 / 55 相比，哨兵复位之后还剩多少失败（这才是没有被污染放大的真实失败数）。若没有测试被点名而整体仍大量失败，把 `01_pytest` 第一处失败之前最近一个失败的完整报错带回。

---

### L0–L6 最终结果表，LIBERO（10 步）：D 组之后，一个 session，整组重跑，按顺序连续跑

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
要的数：每行 `infer` 的 `P10 / P50 / P90 / n`，以及打印的 `effective_config`（必须是 `text_trim=True vae_encoder=native vae_graph=True use_fa4=True use_fa4_mot=True fa4_fallback_reason=None`；有回退这一行作废）。`fp16` 行：`0921_final` 测到 274.70 ms、与未裁剪几乎相同（ISSUE-088，D 组在查原因）。它是一个真实测得的数，**不再作为停止条件**：照常记录，行的备注写 "fp16 does not speed up under text_trim, ISSUE-088"；`effective_config` 必须是完整默认形态，有回退这一行才作废。

**L3 三个精度的精度指标**（真实 LIBERO 数据；`profile default`，同一 session；这是 `imagewam_e2e_official_compare.py` 的口径：vs official 的余弦与 MAE，它自己的延迟列不用于表）
```
SUITE=libero_spatial PRECS="fp16 nvfp4" PROFILES="default" TAG=L3 bash scripts/imagewam_thor_matrix.sh
CALIBRATION=$CAL_TRIM SUITE=libero_spatial PRECS="fp8_static_cutlass" PROFILES="default" TAG=L3_fp8 bash scripts/imagewam_thor_matrix.sh
```
要的数：`matrix_libero_spatial_*_L3*.md` 里每行的 vs official min / median、MAE vs GT median、样本数（`N_TASKS x FRAMES x SEEDS`）。判据：`rc=0`，`FA4 fallback` 为 `None`。

**L4 int8 / int4**（SM80 CUTLASS 的合成基准：GEMM 用随机 packed 操作数、无激活量化，VAE 是替身编码器，整层其余数学是真的；是上界，不是全流程数字；两个各自一个进程）。这两个脚本现在按 workload 和有效 token 数配置：默认 `--workload libero`、`--valid-tokens 24`（序列 `x0=25`，与 FlashRT 各行的裁剪序列相同）、10 步。
```
python benchmarks/imagewam_thor_int8_bench.py --workload libero 2>&1 | tee $OUT/L4_int8.log
python benchmarks/imagewam_thor_int4_bench.py --workload libero 2>&1 | tee $OUT/L4_int4.log
```
要的数：每个日志末尾的 `__INT_BENCH__ {...}` 一行（`prefill_p50_ms`、`denoise_step_p50_ms`、`num_steps`、`p50_ms` = prefill + 步数 × 单步、`x0/a0/total`）。如果报缺 kernel，说明扩展没带 `ENABLE_SM80_INT8_CUTLASS` 构建，把报错带回，不要静默重编。

**L5 收尾复测（夹住 session 内漂移）**：把 L1 与 fp4 的 L2 各再跑一次，日志名 `L5_official.log`、`L5_nvfp4.log`。判据：两次的 P50 与开头的差在 1–2% 内，才认为同一 session 内可比；差更大就在报告里写出来。

**L6 带回什么**：`SESSION`、`P0_clock.log`、`P0_commit.log`、GPU 是否独占；L1/L2/L5 的 P10/P50/P90/n；L2 三行的 `effective_config`；L3 的表；L4 的两个 P50；官方与 FlashRT 用的 checkpoint 的 sha256 前 16 位（`sha256sum $CKPT_PATH | cut -c1-16`）。由有凭据的一方录进 `docs/imagewam_results.json`（先 `check` 再 `render`）。

---

## 已完成的轮次（不再重跑）

逐轮结论已按"用法"第 2、3 条落库，不在本清单重复：每轮做了什么、数字是多少、口径是什么，看 `THOR_STATUS_SUMMARY.md` 的同名轮次小节（`eccf14f`、`a84916a`／`0919e`、`0920`、`0920s4`、`0920t`、`0920c`、`0921`），各项结论看 `opportunities.md` 对应 OPT 条目。逐字的原始记录看 `git log`；本清单只保留"还没做"的东西。

---

## 已决定、不在这里测的

`text_trim` 转默认（原 E1）已经决定并落库：`default` profile 带 `text_trim=True`，新增 `native` profile；门禁与矩阵的默认口径跟着服务默认走（FA4 mot 与原生 VAE 进图后来也提升进了 `default`，见 N 节）；这些结论在 `plan.md` 的 "Decisions pending"、`opportunities.md` OPT-019/030 与 `THOR_STATUS_SUMMARY.md`。

三项 owner 决定（默认提升为最快配置、延迟基线按配置、标定 identity 加入相机几何）已落库，见 `plan.md` 的 "Decisions pending" 与 F1–F3；它们的 Thor 观察已在 `0921` 一轮完成。`plan.md` 的 "Open" 现在只剩目标工作负载自身未确认的声明（指令 token 数是否就是 128、checkpoint / 标定文件 / 图显存预算），它不是 Thor 测试项。

另一份 plan（"Plan: ActionDiT small-M CUTLASS tile selection"，roadmap item 1）还有一个面向 Thor 的相位是 `blocked`：`gemm_variant_autotune` 的 tile 扫描与 `infer()` A/B 从未在 Thor 上跑过（`benchmarks/imagewam_thor_small_m_tile_sweep.py`，`issues.md` ISSUE-023，`opportunities.md` OPT-018）。它不属于本清单的收尾范围（`fast`/`default` 都不开这个开关），列在这里只是为了不让人以为机器上的活已经全干完——如果要做，那是一个独立的短项。
