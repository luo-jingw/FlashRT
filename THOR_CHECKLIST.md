# Thor 测试清单

本节次（配置整合 + `text_trim` 服务化）剩余的待测项。已完成的项已按"用法"第 2 条删除，结论在下面表格指定的文件里。

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

环境变量与构建见 `scripts/imagewam_thor_validation.sh` 文件头。自上一轮起代码改动包含 VAE 几何（`vae_stage.py`、`vae_encoder.py`、前端），仍是 Python，**不含 kernel 与 C++ 改动，不需要重编**；ABI 行需要 `exec/`，native 行需要 `runtime/` 与 `flashrt_imagewam_native`。

```
export OUT=$HOME/thor_val/$(date +%m%d)
python -c "from flash_rt.hardware.jetson_clock_state import report_jetson_clock_state as r; r()" | tee $OUT/P0_clock.log
git rev-parse HEAD | tee $OUT/P0_commit.log
(cd $BUNDLE && sha256sum -c SHA256SUMS) | tee $OUT/P0_bundle.log
```

记录：commit hash；GPU 是否被其他进程占用；`emc_locked` 是否为 `null`（目前未锁定）。这三项写在本轮所有数字旁边。

---

## 已完成（不再重跑）

- `eccf14f`：LIBERO nvfp4 阶梯（spatial 三次重复、goal、libero_10）、叠满精度表、gate nvfp4 202.2 ms、A2 fp16（306.6 → 234.9 → 222.9 ms）、B2（AWQ 0.99970 / 真实 FP8 校准 0.99997）、E2 同口径基线与会话内极差（default 0.4 ms、stack 0.5 ms，`latency_baselines.json` 已按 202.2 ms 重定）、E3 fixture v2 与 trim gate（vs official 0.99931 / min 0.99898，P50 114.6 ms）、S1 三组 pytest、LIBERO 三路径（infer 202.3 / ABI 184.2 / native 183.8；`fast` 93.2 / 95.1 / 跳过）。
- `0919e`：A1 三项拆开定位、B1 两条 gate（e0m3 0.99781 / fp8_static_cutlass 0.99830，都过）、B3 每长度显存（首图 +218.0 MiB reserved / +206.3 allocated，之后 ~0；缓存 32 ≈ 221 MiB）、P 目标三路径（default 216.93 / 173.55 / 173.27；fast 137.64 / 139.35 / 跳过）与 LIBERO 保真逐值相同、S1 驱逐实测、E2 fixture v2 manifest。


- `0920`：**R 四条全绿**（`capture_sync` 1 passed、AWQ 6 passed 且 `plain=285 awq=285`、FA4 开 graph-safety 4 passed 且恢复后 `equal=True cosine=1 max_abs=0`、`model_runtime_vae` 2 passed）→ ISSUE-085 已关闭，ISSUE-080 条件 2 满足。**E2 的 v2 manifest 已按原字节入库**（sha256 `a69a86ac…`，10954 字节，`test_committed_fixture_manifests_are_well_formed` 通过）。

- `0920s4`：**S4 全绿**——native model runtime 按长度 adopt 图（`x0=6`/`14` 两个长度 tick 与 `infer()` `array_equal`、`max_abs=0`），manifest 记 `text_lengths`，`set_text_length` 对没有图的长度回 `-2`；未裁剪一 key 路径整文件过、节点数不变（native 5324 / Python 5348）；schema gate 7 条记录与 golden 逐行相同；parity 两种 `--graph` 全绿、六个 mutant 全检出。数字进 `opportunities.md` OPT-029 与 `THOR_STATUS_SUMMARY.md`。

数字与结论在 `opportunities.md`、`THOR_STATUS_SUMMARY.md`、`issues.md`（ISSUE-080/085/086）。

---

## T. 服务默认变了：重测它的数字（FA4 自动 + 门禁默认跟服务默认）

这一轮两处默认变了，所以**当前记录的服务默认数字（trim-only、门禁 v1/未裁剪那条）不再描述服务默认**：

- `FLASHRT_THOR_FA4` 的默认从 `"0"` 改成 `"1"`：`use_fa4=None`（`default` profile 的值）现在是"这台机器能跑 FA4 就用，否则走 cuBLAS 链"，`FLASHRT_THOR_FA4=0` 强制回退，显式 `use_fa4` 仍然优先。ActionDiT 位点（`use_fa4_mot`）仍是关。
- 门禁（`tests/gate_imagewam_libero.py`）默认改成 fixture **v2（裁剪参考）+ 裁剪开**；未裁剪参考要用 `--no-text-trim --manifest ..._v1.manifest.json`。`scripts/imagewam_thor_validation.sh` 的旧行已显式写成未裁剪，保持原记录含义。

前置：把 `imagewam_libero_gate_v2` 的 fixture 目录放进 `$BUNDLE`（现在 bundle 里只有 v1）。

另外注意：矩阵的开关行现在**每行都显式写 FA4**（`default`/`vae`/`vae_trim` 补了 `FLASHRT_THOR_FA4=0`），因为 FA4 默认已变成"能跑就开"。所以 C 节阶梯里的 `vae_trim` → `vae_trim_fa4bb` 这一步仍然是 FA4 的隔离边际；已记录的 `c20f3a0`/`eccf14f` 数字当时 unset 就等于关，仍然有效。

```
# 1) 服务默认的门禁（不带任何 flag）：nvfp4 与 fp16 各一次
python tests/gate_imagewam_libero.py --precision nvfp4 --fixture-dir "$BUNDLE/imagewam_libero_gate_v2" 2>&1 | tee $OUT/T_gate_nvfp4.log
python tests/gate_imagewam_libero.py --precision fp16  --fixture-dir "$BUNDLE/imagewam_libero_gate_v2" 2>&1 | tee $OUT/T_gate_fp16.log
# 2) 未裁剪参考（旧记录的含义，现在带 FA4）
python tests/gate_imagewam_libero.py --precision nvfp4 --no-text-trim \
  --manifest tests/fixtures/imagewam_gate/imagewam_libero_gate_v1.manifest.json \
  --fixture-dir "$BUNDLE/imagewam_libero_gate_v1" 2>&1 | tee $OUT/T_gate_v1.log
# 3) 端到端与三路径（effective_config 应打印 use_fa4=True）
PRECISION=nvfp4 N_TASKS=10 FRAMES=0,60 SEEDS=0,1 python benchmarks/imagewam_e2e_official_compare.py 2>&1 | tee $OUT/T_e2e_default.log
python benchmarks/imagewam_thor_path_bench.py 2>&1 | tee $OUT/T_path_default.log
python benchmarks/imagewam_thor_path_bench.py --profile fast --precapture 2>&1 | tee $OUT/T_path_fast.log
python benchmarks/imagewam_thor_path_bench.py --profile native 2>&1 | tee $OUT/T_path_native.log
# 4) A/B 归因：同样的命令加 FLASHRT_THOR_FA4=0（关的那条腿现在必须显式给）
FLASHRT_THOR_FA4=0 PRECISION=nvfp4 N_TASKS=10 FRAMES=0,60 SEEDS=0,1 python benchmarks/imagewam_e2e_official_compare.py 2>&1 | tee $OUT/T_e2e_fa4off.log
```

判据：三条 gate 都 pass（服务默认那条对 v2 的 fp16 参考、未裁剪那条对 v1）；`effective_config` 打印 `use_fa4=True`（服务默认）与 `use_fa4=False`（`FLASHRT_THOR_FA4=0`）；P50 与 vs official 记进 OPT-019 / OPT-030；`--profile native` 现在也是**裁剪**集合（只把 FA4 显式关掉），所以它不该再被 R5 跳过。**不设阈值、不判定**。延迟基线 `latency_baselines.json`（202.2 ms，未裁剪、FA4 关）仍是单边界，若要把基线改成描述服务默认，用第 1 条的数再说。
去向：`opportunities.md` OPT-019/OPT-028/OPT-029/OPT-030、`THOR_STATUS_SUMMARY.md`。

---

## S4-pipeline. native 自录图也按长度（本轮新增，需重编）

上一轮的 S4 只做了"采纳前端每长度图"那条；这一轮 native **pipeline 自己录的图**也按长度了，所以 C++ 又变了，**必须重编** `flashrt_imagewam_native`。

```
cmake --build build -j --target flashrt_imagewam_native
IMAGEWAM_NATIVE_PRECISION=nvfp4 python -m pytest tests/test_imagewam_native_pipeline.py tests/test_imagewam_native_runtime.py -q -s 2>&1 | tee $OUT/S4p_native.log
IMAGEWAM_NATIVE_PRECISION=nvfp4 python -m pytest tests/test_imagewam_text_trim_consumer_guards.py -q 2>&1 | tee $OUT/S4p_guards.log
python tests/gate_imagewam_native_parity.py --precision nvfp4 --graph native --bench-iters 50 2>&1 | tee $OUT/S4p_parity_native.log
python tests/gate_imagewam_native_schema_parity.py --precision nvfp4 2>&1 | tee $OUT/S4p_schema.log
```

判据：`test_pipeline_records_one_graph_per_text_length` 两个长度（`x0=6` 先、`14` 后）由 handle 自己装管线并录图，`graph_producer=native`，manifest `text_lengths={'default_key': 14, 'keys': [6, 14], 'per_prompt_length': True}`，两个长度的 tick 都 `array_equal` 到 `infer()`（`actions max_abs=0`）；未裁剪一 key 路径逐行不变（节点数仍 native 5324 / Python 5348、`test_set_pipeline_drops_the_captured_graph` 的 `graph_exec=0 graph_nodes=0 graph_producer=''` 不变）；guards 里那条 GPU 行打印 x0=6 与 x0=10 的 dims / AdaLN 行数 / RoPE 指针随活动长度变化而 buffers 相同；parity `--graph native` 六个 mutant 全检出、节点数不变；schema gate 7 条记录与 golden 逐行相同。
去向：`opportunities.md` OPT-029、`THOR_STATUS_SUMMARY.md`。

---

## E. 收尾


### E1 `text_trim` 转默认 —— 已决定（(c)）

`default` profile 现在带 `text_trim=True`，FA4 与原生 VAE 留在 `fast`；新增 `native` profile（不 trim、FA4 显式关、torch VAE 图外，内容等于旧的 `default`），给 S4 之前的 ABI/native 调用按名字切换，避免踩 R5。改 profile 是 plan 编辑，已记入 `plan.md` 的 "Decisions pending"。

配套已做/已记：矩阵脚本的开关行现在显式写 `TEXT_TRIM`（否则 `default` 行会悄悄变成 `vae_trim` 行，阶梯失去意义）；门禁自己的默认口径仍是"未裁剪参考配置"，要门禁服务默认就是 `--text-trim --manifest ...v2`（那条已经跑过并通过，P50 远低于未裁剪的延迟基线，不需要重测）。

**本轮没有新的 Thor 项。** S4（native 按长度带图）落地后要加的 Thor 项：native 多长度 tick 逐位一致，以及门禁默认口径若改成服务默认，再跑一次 `--text-trim --manifest ...v2` 记录数字。
