# research/

FlashRT 相关的调研与实验材料。每个研究方向（topic）一个目录。

## 从这里开始

主线：一块 Jetson AGX Thor 上，长时程规划器（HF 官方 Qwen3.8-27B）与周期执行器（约 4B 的 VLA/WAM）异步共置；核心问题是规划器的决策延迟 L 怎样影响任务质量（SR、完成时间）。

| 想看什么 | 去哪里 |
|---|---|
| 当前任务单（RoboCerebra 上 MemER 对 HPE） | [long-horizon-planner-executor/tasks_robocerebra.md](topics/long-horizon-planner-executor/tasks_robocerebra.md) |
| 调研方向一：多帧上下文下规划器单次调用怎么变快 | [planner-call-latency](topics/planner-call-latency/README.md) |
| 调研方向二：Thor 上两模型的推理速度与调度 | [thor-two-model-scheduling](topics/thor-two-model-scheduling/README.md) |
| 导览（经典论文、harness 选型、Thor 硬件与实测、排序指标） | [edge-colocation-scheduling](topics/edge-colocation-scheduling/README.md) 下的 `overview.html`（翻页版）与 `overview_page.html`（长页版）；GitHub 不渲染 HTML，需下载后用浏览器打开 |
| 规划器–执行器运行时（harness） | [long-horizon-planner-executor/harness](topics/long-horizon-planner-executor/harness/README.md) |
| 实验与调研的写法（六项说明） | `.claude/skills/brief/SKILL.md`（在 main 分支的仓库根目录） |

## Topics

| Topic | 状态 | 范围 |
|---|---|---|
| [wam-inference-acceleration](topics/wam-inference-acceleration/README.md) | 实验与报告 · 2026-10 至 2026-11 | ImageWAM 推理加速的实验与报告；VLA / WAM 架构与推理剖析调研；机器人工作负载调研；规划器 + 执行器同卡共置（RTX 5090、Jetson AGX Thor） |
| [long-horizon-planner-executor](topics/long-horizon-planner-executor/README.md) | 主线 · 第一轮综述 2026-11-09；仿真闭环 2026-11-10；任务单、harness、Thor 实测 2026-11-19 | 长时程任务为什么用规划器 + 执行器；Jetson AGX Thor 上两模型共置的挑战与实测；RoboCerebra 上的 MemER 对 HPE 任务单与运行时 |
| [planner-call-latency](topics/planner-call-latency/README.md) | 调研方向一 · 立项与论文卡片 7 篇 · 2026-11-19 | 多帧上下文下规划器单次调用的加速：跨调用复用、视觉 token 压缩、混合模型（线性注意力）的缓存、输出侧 |
| [thor-two-model-scheduling](topics/thor-two-model-scheduling/README.md) | 调研方向二 · 立项与论文卡片 9 篇 · 2026-11-19 | Thor 上两模型的推理速度与调度：Apple 同类硬件的系统工作、Jetson 上的 VLA/VLM 实测、快慢双模型系统 |
| [edge-colocation-scheduling](topics/edge-colocation-scheduling/README.md) | 第一轮综述 · 2026-11-17（略读 30 篇，精读 5 篇）；导览页 2026-11-19 | 一块边缘 SoC 上周期截止期模型与 LLM 共置的系统文献：GPU 共享与抢占、LLM 阶段调度、边缘 SoC 实时推理与带宽争用、机器人双系统部署 |

每个 topic 目录下按需包含：

| 文件 | 内容 |
|---|---|
| `README.md` | 范围、状态、文件索引 |
| `survey.md` | 综述：分类框架、各分支脉络、开放问题 |
| `papers.md` | 论文目录，元数据经检索核实 |
| `notes/` | 精读笔记 |
| `reports/` | 报告与汇报材料 |
| `experiments/` | 实验脚本、结果与日志；每个实验的 `README.md` 开头是实验说明的六项（`.claude/skills/brief/SKILL.md`） |

论文 PDF 与文本抽取（各 topic 的 `papers/*/`）只保存在本地，不纳入仓库（`.gitignore`）。
