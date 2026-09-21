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

环境变量与构建见 `scripts/imagewam_thor_validation.sh` 文件头。native 的 C++（pipeline 自己按长度录图）在 `0920t` 那一轮已经编过，此后没有新的 C++ 或 kernel 改动：`43c49ce` 只在 Python 侧（长度表改成 property、native 的测试与门禁各自显式写 `use_fa4=False`）。所以本轮**不需要为了新代码重编**；ABI 行需要 `exec/`，native 行需要 `runtime/` 与 `flashrt_imagewam_native`。

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

- `0920t`：**服务默认（trim + FA4 自动）的三条 gate 全过**——gate nvfp4 对 v2 fixture 0.99889 / 0.99934，P50 **125.86 ms**；gate fp16 对 v2 0.99993 / 0.99997，284.38 ms；未裁剪参考（`--no-text-trim` + v1）0.99418 / 0.99758，191.79 ms（相对旧的 202.2 基线，那条仍只描述"未裁剪 + FA4 关"）。e2e `default` 126.5 ms（served_vs_off 0.99432 / 0.99750），`FLASHRT_THOR_FA4=0` 腿 131.3 ms（0.99441 / 0.99743）——同会话 FA4 开快约 5 ms。三路径（`x0=25`）：`default` 139.47 / 120.29 / 120.05，`fast --precapture` 108.08 / 110.09 / 跳过，`native` 145.93 / 126.58 / 126.38（全部裁剪，native 不再被跳过）。相对 `0919e` 的 114.6 ms 是约 +11 ms 的会话差，不是回归。数字进 `opportunities.md` OPT-019/028/029/030 与 `THOR_STATUS_SUMMARY.md`。

数字与结论在 `opportunities.md`、`THOR_STATUS_SUMMARY.md`、`issues.md`（ISSUE-080/085/086）。

---

## S4-pipeline. native 自录图也按长度（只看一条：双长度自录图）

`0920t` 一轮的结果：未裁剪一 key 路径**全过**（state `array_equal`、`graph_exec=0 graph_nodes=0 graph_producer=''`）、guards 11 passed（GPU 行给出每长度资源表：`x0=6` dims `(6,16,20)`、`x0=10` dims `(10,20,24)`、AdaLN 与 RoPE 随活动长度变、`buffers identical=True`）、parity `--graph native` PASS（native 5324 / Python 5348 节点、六个 mutant 全检出、tick `array_equal` / `max_abs=0`、P50 native 205.27 vs python 207.13）、schema PASS（7 条记录 identical）。
**只有** `test_pipeline_records_one_graph_per_text_length` 红，两条原因已在 `43c49ce` 修掉：`captured_text_lengths` 是 property 但 Protocol 与调用按方法用（`TypeError`），以及该测试拿一边的参考跟另一边的残留行比（`actions` 已经对上，`backbone_hidden`/`K_cache`/`V_cache` 的越界行本来就不同）。修法是两侧跑之前都 `poison_tick_state`，比较保持整块。另：FA4 默认变成"能跑就开"之后，native 路径的测试/门禁各自显式声明 `use_fa4=False`（R6：native pipeline 没有 FA4），不再靠环境变量 `FLASHRT_THOR_FA4=0`；`43c49ce` 也补上了这些显式声明（含 guards 的 `_frontend`，那条 GPU 行会走 `pipeline_resources()`）。下面四条命令重跑本节。

重跑前确认 `flashrt_imagewam_native` 是当前源码编出来的；本轮没有新的 C++ 改动，这条命令只是幂等地保证二进制不旧。

```
cmake --build build -j --target flashrt_imagewam_native
IMAGEWAM_NATIVE_PRECISION=nvfp4 python -m pytest tests/test_imagewam_native_pipeline.py tests/test_imagewam_native_runtime.py -q -s 2>&1 | tee $OUT/S4p_native.log   # 期望 39 passed；不需要 FLASHRT_THOR_FA4
IMAGEWAM_NATIVE_PRECISION=nvfp4 python -m pytest tests/test_imagewam_text_trim_consumer_guards.py -q 2>&1 | tee $OUT/S4p_guards.log
python tests/gate_imagewam_native_parity.py --precision nvfp4 --graph native --bench-iters 50 2>&1 | tee $OUT/S4p_parity_native.log
python tests/gate_imagewam_native_schema_parity.py --precision nvfp4 2>&1 | tee $OUT/S4p_schema.log
```

判据：`test_pipeline_records_one_graph_per_text_length` 两个长度（`x0=6` 先、`14` 后）由 handle 自己装管线并录图，`graph_producer=native`，manifest `text_lengths={'default_key': 14, 'keys': [6, 14], 'per_prompt_length': True}`，两个长度的 tick 都 `array_equal` 到 `infer()`、`differing=[]`、`actions max_abs=0`。两侧的 state 缓冲都是整块比对（`np.array_equal` 全长度），且两边都先 NaN 填过再跑（`_infer_reference` 与 `_poisoned_native_tick` 各在跑之前 `poison_tick_state`），所以一条长度自己的图没写到的行在两侧都停在基线上——哪一侧越界写了自己不拥有的行，或者写得不一样，仍然会红，不再靠"残留 vs NaN"；未裁剪一 key 路径逐行不变（节点数仍 native 5324 / Python 5348、`test_set_pipeline_drops_the_captured_graph` 的 `graph_exec=0 graph_nodes=0 graph_producer=''` 不变）；guards 里那条 GPU 行打印 x0=6 与 x0=10 的 dims / AdaLN 行数 / RoPE 指针随活动长度变化而 buffers 相同；parity `--graph native` 六个 mutant 全检出、节点数不变；schema gate 7 条记录与 golden 逐行相同。
去向：`opportunities.md` OPT-029、`THOR_STATUS_SUMMARY.md`。

---

## 已决定、不在这里测的

`text_trim` 转默认（原 E1）已经决定并落库：`default` profile 带 `text_trim=True`，FA4 与原生 VAE 留在 `fast`，新增 `native` profile；门禁与矩阵的默认口径跟着服务默认走；这些结论在 `plan.md` 的 "Decisions pending"、`opportunities.md` OPT-019/030 与 `THOR_STATUS_SUMMARY.md`。

唯一还等 owner 的延迟口径问题也记在 `plan.md` 的 "Open"：`latency_baselines.json` 的 202.2 ms 描述的是未裁剪、FA4 关的配置，而它现在被用在服务默认上（同一 gate 里服务默认 125.86 ms），是否按服务默认重新定基线、以及文件里是否逐条写明各数字对应的配置，是那次决定的内容。它不是 Thor 测试项。
