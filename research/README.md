# research/

ImageWAM 推理加速的调研与实验材料（RTX 5090 为主，Thor 数据见 `docs/imagewam_thor_profile.md`）。

## 报告

| 文件 | 内容 |
|---|---|
| `reports/report_acceleration_research.md` | 算法与系统层面调研汇总：网格子集采样（含 2 步实测延迟）、Picard、雅可比、表示与 Q/K/V 秩、参考论文、设备状态、方向排序 |
| `reports/report_actiondit_batched_m.md` | 实验 1–9 的完整记录：batched-M 单步成本、Picard 收敛、按组 attention、nvfp4 GEMM tile、量化融合、雅可比、网格子集（开环 / LIBERO / RoboTwin 闭环） |

## 实验数据（`experiments/`）

每个目录放该实验的脚本、结果 JSON 和日志。原始张量（`.npz`，共约 6 GB）和编译产物未收录，可用目录内脚本重新生成。脚本里的输出路径指向当初的临时目录，重跑前需改成本地路径。

| 目录 | 对应 | 内容 |
|---|---|---|
| `e1_e3_batched_m_step/` | 实验 1、3 | M = 64k 单步耗时、按组 FA2 attention、nvfp4 投影、Picard 加速投影 |
| `e2_picard_convergence/` | 实验 2 | 全窗口 / 滑动窗口 Picard 收敛（fp8、nvfp4） |
| `e4_nvfp4_gemm_tiles/` | 实验 4 | CUTLASS SM120 tile / swap-AB / stream-K / split-K 测试源码与结果，GEMM M 扫描 |
| `e5_nvfp4_quant_fusion/` | 实验 5 | nvfp4 量化融合原型源码与结果 |
| `e6_e7_jacobian_fewstep/` | 实验 6、7 | 雅可比与线性度、网格子集开环误差 |
| `e8_libero_closed_loop/` | 实验 8 | LIBERO-10 闭环（串行 10 步 / {0, 7, 9} / {0, 9}） |
| `e9_robotwin_closed_loop/` | 实验 9 | RoboTwin 2.0 闭环逐任务结果 |
| `grid2_latency/` | 网格子集实测 | 服务路径 {0, 9} 两次求值的 `infer()` 延迟（fp8 / nvfp4 / fp16） |
| `actiondit_representation/` | 表示分析 | 各层 token 有效秩、夹爪方向、基重建、跨任务共享基、手臂动作可预测性 |
| `qkv_rank_fixed_instruction/` | Q/K/V 秩 | 固定指令、不同视觉条件下 h 与 Q/K/V 的 k90 / k99 / 单头秩 |
| `libero_e2e_latency/` | 端到端延迟 | 官方权重 + 标准 LIBERO 输入的闭环端到端延迟 |
| `clock_state_5090/` | 设备状态 | 5090 时钟 / 温度随时间与跨进程的延迟 |

已纳入仓库的对应脚本在 `benchmarks/`：`imagewam_batched_m_step_bench.py`、`imagewam_picard_convergence.py`、`imagewam_velocity_jacobian_study.py`、`imagewam_few_step_study.py`、`imagewam_5090_clock_state_probe.py`。

## 论文（`papers/`）

参考论文清单与链接见 `papers/README.md`；PDF 与文本抽取只保存在本地，不纳入仓库。
