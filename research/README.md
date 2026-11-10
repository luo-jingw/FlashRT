# research/

FlashRT 相关的调研与实验材料。每个研究方向（topic）一个目录。

## Topics

| Topic | 状态 | 范围 |
|---|---|---|
| [wam-inference-acceleration](topics/wam-inference-acceleration/README.md) | 实验与报告 · 2026-10 至 2026-11 | ImageWAM 推理加速的实验与报告；VLA / WAM 架构与推理剖析调研；机器人工作负载调研；规划器 + 执行器同卡共置（RTX 5090、Jetson AGX Thor） |
| [long-horizon-planner-executor](topics/long-horizon-planner-executor/README.md) | 第一轮综述 · 2026-11-09 | 长时程任务为什么用规划器 + 执行器；Jetson AGX Thor 上两模型共置的独特挑战与实测发现；目标指标；相关系统论文的写法 |

每个 topic 目录下按需包含：

| 文件 | 内容 |
|---|---|
| `README.md` | 范围、状态、文件索引 |
| `survey.md` | 综述：分类框架、各分支脉络、开放问题 |
| `papers.md` | 论文目录，元数据经检索核实 |
| `notes/` | 精读笔记 |
| `reports/` | 报告与汇报材料 |
| `experiments/` | 实验脚本、结果与日志 |

论文 PDF 与文本抽取（各 topic 的 `papers/*/`）只保存在本地，不纳入仓库（`.gitignore`）。
