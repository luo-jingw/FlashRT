# Thor 测试清单

`0921d`（D 组）与 `0921d_final`（L 组，LIBERO 一张最终表）已做完并落库：表在 `docs/imagewam_results.md`（int8 / int4 两行按决定留空），数字与口径在 `THOR_STATUS_SUMMARY.md` 的 `0921d` 小节。本节次剩下的都是"让表更可信"的小项，不阻塞表；RoboTwin 那张等它的 workload 声明（收集指令已给 Thor 上的 agent），不在本清单。

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

### X1 fp16 的 cuBLASLt GEMM 探针（ISSUE-088；表里 fp16 一行的数字目前不稳）
```
python benchmarks/imagewam_fp16_gemm_probe.py --x0 25,513 2>&1 | tee $OUT/X1_fp16_gemm_probe.log
```
每个 GEMM 形状（`x0=25` 的裁剪长度与 `x0=513` 的未裁剪长度）各测三种算法：启发式第一名、frontend 用的"零填充上自动调优"、"随机数据上自动调优"，都在随机操作数上用 CUDA event 重测，三个新 runner 取 min..max，并给出实测 TFLOPs。要带回：整份日志。读法：哪些形状的 TFLOPs 远低于 CUTLASS 的水平（几十 TFLOPs 以上）；同一形状三个 runner 的 min..max 是否分叉（说明调优结果不稳定）；"零填充"与"随机"两列是否不同（说明数据填充影响选择）。判据不是过 / 不过，是给 ISSUE-088 定位。

### X4 nvfp4 的 24 token 图：计算、访存还是 kernel 延迟（回答"瓶颈在哪"）
D1 里每个精度的第一张图抓到 0 个 kernel（profiler 第一次会话要初始化 CUPTI）；脚本已加一次空会话预热，并新增"平均 kernel 时长与短 kernel 的占比"两行。**不要再开 nsys**（会和 torch.profiler 抢 CUPTI）。
```
for P in nvfp4 fp16_cutlass; do
  python benchmarks/imagewam_graph_kernel_profile.py --precision $P --valid-tokens 24 --use-fa4 off 2>&1 | tee $OUT/X4_$P.log
done
```
要带回：每个日志里 `graph replay ... ; kernels inside ...`、`average kernel ... us; kernels < 10 us: ...`、分类耗时、top 12。读法：短 kernel（< 25 us）占 GPU 时间的比例很大，说明是 kernel 数量 / 延迟受限；GEMM 类占大头而平均 TFLOPs 远低于 110，说明 GEMM 效率问题；top kernel 是逐元素类，说明访存受限。

### X5 计算图 × 算子网格（fp4 为什么只快 4 倍：瓶颈在哪一格）
`benchmarks/imagewam_stage_operator_grid.py`：一次 eager 前向（与图里同样的 kernel、同样的顺序）套 `torch.profiler`，每个阶段（encode、backbone 的 double / single 层、prefill 余项、ActionDiT 的 double / single 层、每步余项）打命名区间，把每个 kernel 归到发射它的最内层阶段和按名字分的算子类（gemm / quantize / norm / rope / attention / glu / residual / copy / other），输出：阶段 × 算子类的 GPU 毫秒网格、每格 kernel 数与平均时长、每个阶段里短于 10 / 25 µs 的 kernel 的时间占比、每个层阶段 GEMM 的实测 TFLOPs 与权重字节流速（GB/s，由模型维度解析推出）。
```
for P in nvfp4 fp16_cutlass; do
  python benchmarks/imagewam_stage_operator_grid.py --precision $P --valid-tokens 24 --use-fa4 off 2>&1 | tee $OUT/X5_$P.log
done
python benchmarks/imagewam_stage_operator_grid.py --precision fp8_static_cutlass --calibration $CAL_TRIM --valid-tokens 24 --use-fa4 off 2>&1 | tee $OUT/X5_fp8.log
python benchmarks/imagewam_stage_operator_grid.py --precision nvfp4 --valid-tokens 24 --use-fa4 on 2>&1 | tee $OUT/X5_nvfp4_fa4.log
```
要带回：四份日志全文（网格、短 kernel 占比、TFLOPs 与 GB/s、"no class matched" 列出的 kernel 名字——这些名字用来补分类）。读法：GEMM 的 TFLOPs 与 GB/s 都远低于硬件（fp16 约 110 TFLOPs、fp8 约 270 TFLOPs 可达；带宽约 250 GB/s）→ 是效率或延迟受限，不是算力或带宽受限；某个阶段的短 kernel 占比高 → 该阶段是 launch / 尾部延迟受限；某一列（quantize / norm / glu / residual）在网格里占大头 → 那一类就是融合的目标。若 profiler 报 "trace holds no GPU kernels"，把报错带回。

### X2 官方那一行的漂移（表里官方 L1 = 456.66 ms，末尾复测 L5 = 377.04 ms，−17.4%）
`emc_locked=null`，官方 eager 更偏访存，FlashRT 的 nvfp4 两端只差 −0.4%，所以怀疑是 EMC 频率没锁。记录 EMC 频率，不改频：
```
tegrastats --interval 500 --logfile $OUT/X2_tegrastats_a.log &   # 后台
python benchmarks/imagewam_official_torch_bench.py --workload libero --warmup 5 --iters 30 2>&1 | tee $OUT/X2_official_a.log
kill %1
# 空闲 5 分钟后再来一遍，日志名 _b
```
要带回：两次官方的 P50，以及两份 tegrastats 日志里 `EMC_FREQ` 的取值范围。两次官方都是 456 左右而 EMC 一样 → 漂移来自别处，把 L5 当时之前跑了什么带回。

### X3 剩下的两个真实测试失败（ISSUE-089）
```
python -m pytest tests/test_imagewam_text_trim.py -x -q -s 2>&1 | tee $OUT/X3_text_trim.log
```
要带回：失败的两个测试的断言原文与两侧数值。

---

## 已完成的轮次（不再重跑）

逐轮结论已按"用法"第 2、3 条落库，不在本清单重复：每轮做了什么、数字是多少、口径是什么，看 `THOR_STATUS_SUMMARY.md` 的同名轮次小节（`eccf14f`、`a84916a`／`0919e`、`0920`、`0920s4`、`0920t`、`0920c`、`0921`），各项结论看 `opportunities.md` 对应 OPT 条目。逐字的原始记录看 `git log`；本清单只保留"还没做"的东西。

---

## 已决定、不在这里测的

`text_trim` 转默认（原 E1）已经决定并落库：`default` profile 带 `text_trim=True`，新增 `native` profile；门禁与矩阵的默认口径跟着服务默认走（FA4 mot 与原生 VAE 进图后来也提升进了 `default`，见 N 节）；这些结论在 `plan.md` 的 "Decisions pending"、`opportunities.md` OPT-019/030 与 `THOR_STATUS_SUMMARY.md`。

三项 owner 决定（默认提升为最快配置、延迟基线按配置、标定 identity 加入相机几何）已落库，见 `plan.md` 的 "Decisions pending" 与 F1–F3；它们的 Thor 观察已在 `0921` 一轮完成。`plan.md` 的 "Open" 现在只剩目标工作负载自身未确认的声明（指令 token 数是否就是 128、checkpoint / 标定文件 / 图显存预算），它不是 Thor 测试项。

另一份 plan（"Plan: ActionDiT small-M CUTLASS tile selection"，roadmap item 1）还有一个面向 Thor 的相位是 `blocked`：`gemm_variant_autotune` 的 tile 扫描与 `infer()` A/B 从未在 Thor 上跑过（`benchmarks/imagewam_thor_small_m_tile_sweep.py`，`issues.md` ISSUE-023，`opportunities.md` OPT-018）。它不属于本清单的收尾范围（`fast`/`default` 都不开这个开关），列在这里只是为了不让人以为机器上的活已经全干完——如果要做，那是一个独立的短项。
