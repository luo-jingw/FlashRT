# 一块边缘 SoC 上周期截止期模型与 LLM 的共置：系统文献综述

- 快照日期：2026-11-17（第一轮）。
- 引用格式：方括号里是 [papers.md](papers.md) 的编号（如 [A05]），后面注明原文位置。
- 精读笔记在 `notes/` 下；论文目录中卡片与笔记不一致的地方，以笔记为准。

## 0. 范围与方法

### 0.1 调研问题

1. 相关系统论文怎样表述问题、优化目标和指标？
2. 它们的核心观察是什么，怎么测出来的？
3. 用了什么机制，和哪些基线比，怎么评估？
4. 它们的前提（硬件、负载、截止期、内存结构、可抢占性）与本场景差在哪里？

本综述只回答这四个问题，不给本场景设定优化目标。

### 0.2 场景参照

下表是判断"文献前提是否成立"时使用的事实。数字来自本组在 Jetson AGX Thor 上的测量轮次（B25–B28，以及 [long-horizon-planner-executor](../long-horizon-planner-executor/README.md) topic 的记录）。

| 项 | 取值 |
|---|---|
| 硬件 | Jetson AGX Thor：Blackwell 集成 GPU，20 个 SM；14 核 Arm CPU；128 GB LPDDR5X 统一内存，CPU 与 GPU 共用约 273 GB/s |
| 带宽实测 | GPU 拷贝 255 GB/s；14 个 CPU 核可达 223 GB/s；两者同时运行，合计只有 GPU 单独运行时的 0.93–0.98 倍 |
| 执行器 | ImageWAM 动作模型；单独运行一次 37.5 ms；周期调用，截止期 200 ms（严格）或 533 ms；一次输出 16 步动作，每 200 ms 消耗 4 步 |
| 规划器 | Qwen3.8-27B VLM（NVFP4）；每 5.33 s 一次 monitor 调用，输入 4–6k token；prefill 1.2–2.2 s，以 2048 token 为一块、每块一次 forward（约 1.17 s），块切得更小会改变输出；decode 约 27 ms/token，回复 14–80 token |
| 可用控制 | MPS 按百分比限 SM（最少 8 SM）；green context（本组实测：请求 4 SM 实得 8 SM）；MPS 客户端优先级（2 档）；CUDA stream 优先级。没有 MIG，没有内存带宽分区 |
| 已测现象 | 不分 SM 时，执行器主要被规划器 prefill 拖慢；按 SM 不重叠划分后，剩余干扰来自 decode 的带宽占用；prefill 更快、更密时执行器更慢；MPS 客户端优先级（不限 SM）下执行器 0 % 错过（p99 70 ms），规划器 p95 2.5 / 5.0 s（wrist-224 / 720）；CPU 内存负载使规划器 decode 慢约 2.2 倍，执行器 p99 翻倍 |

### 0.3 检索与筛选

- **四类**：
  - A. GPU 共享与抢占；
  - B. LLM 服务的阶段调度；
  - C. 边缘 SoC 上的多模型实时推理与内存带宽争用；
  - D. 机器人双系统的部署。
- **来源**：
  - 从起点论文出发，前后向查引用；
  - 会议扫描，2020–2026：OSDI、SOSP、EuroSys、ATC、NSDI、ASPLOS、MLSys、RTSS、RTAS、ECRTS、SenSys、MobiSys、MobiCom、ICCAD、NeurIPS、ICLR；
  - arXiv 预印本、GitHub PR、学位论文。
- **筛选**：
  - 纳入：两个以上负载共用一个加速器，或"慢规划器 + 快执行器"的部署。
  - 排除：只讨论单模型加速的工作，以及集群级放置的工作。
- **数量**：略读 30 篇，见 [papers.md](papers.md)；精读 5 篇，见第 5 节。
- **元数据核对**：
  - 来源：arXiv 页面、会议论文集页面、论文首页的会议印记或 DOI、Crossref。DBLP 在检索期间拒绝自动访问，没有用上。
  - 未核实的条目标 †。
- **检索词、候选清单和排除理由**：见 [search_log.md](search_log.md)。

## 1. 分类框架

四个轴：

1. **受保护方与让步方**：谁的时延必须守住，谁可以让。
2. **控制维度**：
   - 时间：控制下发，或制造抢占点；
   - 空间：分 SM；
   - 带宽：给 CPU 限流；
   - 放弃并发。
3. **硬件**：独立显卡（HBM 归 GPU 独占），或集成 SoC（CPU 与 GPU 共用 LPDDR）。
4. **时限形式**：统计 SLO（P99 或达成率）、可调度性分析（周期加截止期），或只报均值。

| 编号 | 工作 | 受保护方 / 让步方 | 控制维度 | 硬件 | 时限形式 |
|---|---|---|---|---|---|
| A01 | REEF | 固定频率小 DNN / 闭环小 DNN | 时间（重置抢占）+ 空间（kernel padding） | 独显 | 时延近独占 |
| A02 | Orion | HP 推理或训练 / BE | 空间 + 按计算或访存特征配对 | 独显 | P99 近独占 |
| A03 | Clockwork | 各请求 / batch 请求 | 放弃并发 | 独显 | 请求级 SLO |
| A04 | Paella | 多客户端 | 时间（软件逐 kernel 下发） | 独显 | P99 |
| A05 | XSched | 前台 / 后台（含 SoC 上跨引擎） | 时间（三级抢占） | 独显 + Jetson Orin | P99 倍数 |
| A06 | LithOS | HP 推理（含 LLM）/ BE 训练 | 空间（TPC mask）+ kernel 切分 | 独显 | MLPerf 时延约束 |
| A07 | GCAPS | 周期实时任务 / BE | 时间（驱动 TSG 抢占） | Jetson Xavier NX / Orin Nano | 可调度性 |
| A08 | BLESS | 按配额的多应用 | 空间（预建 MPS context）+ 时间 | 独显 | 配额 |
| A09 | Hummingbird | LLM serving / 离线 LLM 或训练 | 时间（block 级切分） | 独显 | P99 TTFT / TPOT |
| B10 | DistServe | 同一模型的 decode / prefill | 分到不同 GPU | 独显集群 | TTFT / TPOT 达成率 |
| B11 | Sarathi-Serve | decode / prefill | 时间（chunked prefill + token budget） | 独显 | TBT P99 |
| B12 | semi-PD | decode / prefill | 空间（MPS 百分比） | 独显 | TTFT / TPOT |
| B13 | MuxWise（Drift） | decode / prefill | 空间（green context）+ 按层下发 | 独显 | P99 TBT / TTFT |
| B14 | Nexus | decode / prefill | 空间（green context） | 独显 | TTFT / TBT |
| B15 | Bullet | decode / prefill | 空间（MPS + SM mask） | 独显 | P90 TTFT / TPOT |
| B16 | DuetServe | decode / prefill | 空间（TPC mask，按需开启） | 独显 | TBT |
| C17 | Miriam | 关键 DNN / 普通 DNN | 空间（elastic kernel） | RTX 2060 / Xavier | 时延近独占 |
| C18 | RT-mDL | 周期 DNN 任务 | 串行 + 模型缩放 | Jetson Xavier 等 | 截止期错过率 |
| C19 | Twill | DNN 与 transformer 混合 | 映射 + 冻结 + DVFS | Jetson Orin NX | 功耗预算 |
| C20 | Martín et al. | 两个对称推理进程 | 空间（MPS / green context / MIG）对比 | A100 + Jetson Orin | 超时比例 |
| C21 | BWLOCK++ | GPU 实时 kernel / CPU 任务 | 带宽（CPU 限流） | Jetson TX2 | 最坏 slowdown |
| C22 | 动态带宽分配 | GPU 实时 kernel / CPU 任务 | 带宽（按进度放宽限流） | Jetson TX2 | WCET |
| C23 | Serafini 第 6 章 | CPU 关键任务 / GPU kernel | 空间（用 green context 限 SM 来限带宽） | Jetson AGX Orin / Thor | slowdown |
| D24 | TRT-Edge-LLM PR #193 | 轨迹模型 / 7B VLM 规划器 | stream 优先级（同进程） | Jetson Thor | 平均与最坏 tick |
| D25 | DualVLN on Thor | System 1 / System 2 | 运行时设置 | Jetson Thor | 观测年龄 p95 |
| D26 | NavGPT-3 | 安全监视 / 规划器与 VLA | 软件层控制权转移 | Thor 上的 VLA + 远端规划器 | halt p95 |
| D27 | OxyGen | 动作 / 语言（同一模型） | 同一进程串行 + 合批 | RTX 4090 / Thor | 平均频率 |
| D28 | FiS-VLA | 快系统 / 慢系统（参数共享） | 训练时异步采样 | RTX 4090 | 平均频率 |
| D29 | RTC | 执行器 / 远端推理 | 执行器侧动作块 inpainting | 远端 RTX 4090 | 推理延迟作为 deadline d |
| D30 | SmolVLA | 客户端 / 远端策略 | 动作队列阈值 | 远端 | 任务完成时间 |

## 2. 各类脉络

### 2.1 A. GPU 共享与抢占

- **问题表述**：一块 GPU 上有一个高优先级任务和若干可让步任务。目标是"高优先级时延接近独占，同时最大化低优先级吞吐或总利用率"。BLESS 是例外，按配额做公平。只有 GCAPS [A07] 使用周期、截止期和响应时间分析。
- **约束**：NVIDIA GPU 不支持主动抢占已经在运行的 kernel，各工作对这一点的描述一致。因此机制的主线是"控制让步方占用设备的时长"：
  - **限制在途命令**：REEF-N 只清空排队中的 kernel [A01 §4.4]；Orion 在途 BE 时长不超过 HP 时延的 2.5 % [A02 §5.1]；XSched Lv1 的在途阈值使抢占延迟约为阈值 × 命令时长 [A05 Fig.11a]；Paella 逐 kernel 下发 [A04 §6]。
  - **撤回或中断**：REEF 在 AMD 上杀掉幂等 kernel，抢占延迟 35–38 µs [A01 Fig.12b]；XSched Lv2 改写 SASS，Lv3 用 TSG 时间片或未公开 ioctl [A05 §6]；GCAPS 修改驱动 runlist [A07 §5]。
  - **切小让步方的 kernel**：LithOS 的 atomizer [A06 §7]；Hummingbird 按 thread block 切成一波大小，平均每段 77 µs，平均抢占延迟 139 µs [A09 §4.2、Fig.13a]。
  - **空间**：stream 优先级、MPS 限 SM、TPC mask [A06]、预建多个 SM 受限的 context [A08 §4.5]、按计算或访存特征配对 kernel [A02]。
  - **放弃并发换可预测性**：Clockwork 让每个 GPU 一次只跑一个推理 [A03]。它的依据是，用线程池并发跑推理，吞吐最多 +25 %，尾时延却升约 100 倍 [A03 §2、Fig.2b]。
- **带宽残余干扰**：多篇独立测到"计算资源隔离之后仍有带宽干扰"，见 3.4 节。没有一篇在机制上做内存带宽隔离；Hummingbird、BLESS、XSched 都把它列为空间共享解决不了的部分。
- **与本场景差别最大的前提**：
  - 硬件：几乎都在独立显卡上评估。
  - 让步方：多为训练或离线批处理，没有自己的时延要求。
  - 让步方的不可抢占单元：在 µs 到 ms 量级。最长的是 LithOS 报告的 DLRM 训练 kernel，超过 30 ms [A06 §5]；Hummingbird 报告的 LLM GEMM 达 7.49 ms [A09 §1]。

### 2.2 B. LLM 服务的阶段调度

- **问题表述**：同一模型的 prefill（计算密集）和 decode（访存密集）共置时互相干扰。
  - 指标是 TTFT 和 TBT（或 TPOT）的 P90 / P99，加上达成率；或者用"满足 SLO 的最大请求率"（goodput）。
  - SLO 一般定义为无干扰时延的若干倍：Sarathi 5× / 25×，semi-PD 7.5× / 10×。
- **约束干扰的三条路**：
  - **时间**：Sarathi-Serve 用 token budget 限制每轮的 prefill 量 [B11 §4.3]。代价是：1k 块 chunked prefill 的计算效率从 71 % 降到 61 %，总 prefill 时延为不切块的 1.13 倍 [B15 §2.3.2]。MuxWise 按 transformer 层下发 prefill，在层边界抢占 [B13 §3.2.3、§3.4]。
  - **空间（同一 GPU 内分 SM）**：
    - semi-PD 用 MPS 百分比，调整比例要重启进程 [B12 §5]；
    - MuxWise 和 Nexus 用 green context，所有分区在启动时预建，运行时只切换 [B13、B14 §5]；
    - Bullet 在 MPS 下用 libsmctrl 设 SM mask，重配置约 4.1 µs [B15 Table 5]；
    - DuetServe 用 TPC mask，只在预测会违反 TBT 时才开启分区 [B16 §4]。
  - **分离到不同 GPU**：DistServe [B10]。它论证"同卡上分开排队、优先任一阶段都会损害另一阶段" [B10 §2.3]。
- **带宽争用的结论不一致**（见 3.4 节）。这些测量都在 HBM 上做，最大的数字也只到约 36 %。
- **与本场景差别最大的前提**：
  - 受保护方始终是同一模型的 decode。prefill 一方虽然让步，但有 TTFT 要求，这一点与本场景"让步方也有时延要求"同形。
  - SM 规模为 78–132 个，分区粒度 2–16 SM。
  - 文献中的切分（token 块或层）都假定不改变数值，但都没有做输出一致性测试。

### 2.3 C. 边缘 SoC 上的多模型实时推理与内存带宽争用

- **C1 多 DNN 实时调度**：负载都是毫秒级 CNN 或小 transformer。
  - RT-mDL [C18] 有周期任务和错过率指标，用 mutex 让 GPU 子任务串行执行，截止期设为平均执行时间的 4 或 6 倍。
  - Miriam [C17] 用 source-to-source 变换得到 elastic kernel，把普通任务填进关键 kernel 剩下的 SM。它的前提是"边缘 GPU 上没有 MPS"，这在 Thor 上不成立。
  - Twill [C19] 一个 cluster 同时只跑一个模型。
  - Martín et al. [C20] 在 Orin 上直接比较了 MPS 与 green context，结论是：
    - green context 不隔离内存；
    - Orin Nano 受功耗限制时 GPU 降频，隔离更差；
    - Orin AGX 固定 GPU 频率后，隔离接近 A100 上的 MIG。
- **C2 带宽调节**：
  - BWLOCK++ [C21] 和动态带宽分配 [C22] 只处理"CPU 拖慢 GPU"这一个方向：用 PMC 按 1 ms 周期给 CPU 核限流，并假设 GPU 上一次只有一个实时 kernel。
  - Serafini 第 6 章 [C23] 是唯一在 Thor 上测量的工作：
    - 方向相反，是 GPU 拖慢 CPU；
    - GPU 合并写约 4 个 SM 就让 CPU slowdown 达到最大值 2.5×；
    - 512 B 跨步写只用 1–2 个 SM 就把 CPU 拖慢约 4.5–5×；
    - 用 green context 减少 SM，对合并访存有效，对跨步写无效。
  - 本类文献中没有调节两个 GPU 负载之间带宽的机制。
- **与本场景差别最大的前提**：受保护方和干扰方要么都是小 DNN，要么是 CPU 与 GPU 之间。本场景是 GPU 上的 LLM 干扰 GPU 上的执行器，同时叠加 CPU 内存负载，三方争用同一份带宽。

### 2.4 D. 机器人双系统的部署

- **部署位置分三种**：
  - 两块独立 GPU：Helix，未收入目录，见 search_log.md；
  - 规划器在远端：NavGPT-3 [D26]、RTC [D29]、SmolVLA [D30]；
  - 同一设备：PR #193 [D24]、DualVLN on Thor [D25]、OxyGen [D27]、FiS-VLA [D28]。
- **同一设备的工作里测了两模型干扰的只有 PR #193**：
  - 规划器是 Qwen2.5-VL-7B，prefill 1024 token、约 75 ms。
  - 执行器每条轨迹耗时：
    - 规划器空闲时 48.1 ms；
    - 同进程，执行器用高优先级 stream：55.9 ms；
    - 跨进程，不开 MPS：117–122 ms。
  - 异步时最坏 tick 78 ms，阻塞时 331 ms。
- **运行方式几乎都是异步**：执行器读最新的规划，staleness 被当作设计的一部分。
- **指标以均值为主**：Hz、中位时延、任务时间。报尾部的只有三处：NavGPT-3 的 halt p95、DualVLN on Thor 的观测年龄 p95、PR #193 的最坏 tick。没有一篇统计执行器的截止期错过率。
- **问题表述最接近"约束一方、优化另一方"的是 OxyGen**：
  - 式 3：max τ = Bk/T，s.t. f = H/T ≥ f_min [D27 §3.1]。
  - 但 v5 没有给出 f_min 的取值，实验中也没有检验这个约束。
  - 它在 Thor 上的 MPS 两进程并行比顺序执行慢约 15–25 %，这个数只在图中出现 [D27 §4.2、Fig.4、Fig.6]。

## 3. 跨类综合

### 3.1 问题表述的三种模板

| 模板 | 形式 | 代表 |
|---|---|---|
| 保护一方、收割空闲 | 高优先级时延接近独占（P99 倍数或 SLO 达成率），最大化低优先级吞吐或利用率 | A01、A02、A05、A06、A09、C17 |
| 统计 SLO 下的容量 | 满足 TTFT / TBT 的 P90 / P99 前提下，最大请求率或 goodput | B10–B16、A03 |
| 约束一方频率、最大化另一方吞吐 | max τ s.t. f ≥ f_min | D27 |

实时系统方向的 GCAPS [A07] 和 RT-mDL [C18] 用的是可调度性或错过率，负载都是小任务。

### 3.2 指标

| 指标 | 用在哪里 | 怎么定义 |
|---|---|---|
| 高优先级 P99 相对独占的倍数 | A01、A02、A05 | 单独运行为 1 |
| SLO 达成率 | A06、A09、B 类 | SLO = 独占 P99 或其若干倍 |
| goodput / 容量 | A03、B10、B13 | 满足 SLO 的最大请求率 |
| 抢占延迟 | A01、A05、A09 | 从高优先级到达到开始执行，µs 级 |
| 可调度性、WCRT、错过率 | A07、C18、C20 | 周期任务集 |
| 平均频率 | D24、D27、D28 | H / 平均帧时延 |
| 观测年龄、halt 时延的 p95 | D25、D26 | 决策所依据的观测有多旧；从触发到停车 |

### 3.3 机制谱系与在 Thor 上的条件

| 机制 | 代表 | 依赖什么 | 在本场景能否成立 |
|---|---|---|---|
| stream 优先级 | A02、D24 | 只在同一 context 内有效 | 只适用于两个模型在同一进程内；D24 测得执行器仍慢约 16 % |
| MPS 按百分比限 SM | B12、C20 | MPS；改比例要重启 | 可用；Thor 上最少 8 SM |
| MPS 客户端优先级 | 无 | MPS | 本轮的 30 篇中没有一篇评估过 |
| 预建多个 SM 受限的 context | A08、B13、B14 | green context 或 cuCtxCreate_v3；CUDA Graph 每个分区要各捕获一套 | 可用；20 SM 下可选的分区很少，粒度待核实（见第 6 节） |
| TPC mask（libsmctrl） | A06、B15、B16 | 逆向得到的接口 | Thor 上是否可用，文献中没有证据 |
| 限制在途命令 | A04、A05 Lv1 | 拦截公开的 driver API | 一个 CUDA graph 只算一条命令 [A05 §9]，graph 运行期间无法抢占 |
| block 级切分 kernel | A06、A09 | PTX 改写；cuBLAS / cuDNN 不支持；CUDA graph 未评估 | 规划器的 kernel 是否带 PTX 未知；数值是否一致论文没有讨论 |
| 按层下发 | B13 | 框架层改写下发逻辑 | 不改变 token 块的大小；输出是否一致没有实测 |
| 撤回或中断 kernel（Lv2 / Lv3、TSG） | A05、A07 | SASS 插桩、未公开 ioctl 或改驱动；TSG 只在进程之间有效 | 论文只覆盖 Kepler / Volta / Ampere；TSG 与 MPS 不兼容 |
| 计算 / 访存配对 | A02 | 离线给 kernel 打标签 | 在带宽受限的 SoC 上，A09 §3 指出这种标签太粗 |
| CPU 带宽限流 | C21、C22 | PMC + 内核模块 | 只处理 CPU 拖慢 GPU；Thor 的 CPU 是 Neoverse-V3AE，文献用的是 TX2 |
| 用 green context 限 SM 来限带宽 | C23 | green context | 对合并访存有效，对跨步写无效 |
| 放弃并发 | A03 | 无 | 让步方的单元约 1.17 s，远大于 200 ms 的周期 |

### 3.4 带宽干扰：已有测量

| 工作 | 平台 | 谁干扰谁 | 结果 | 位置 |
|---|---|---|---|---|
| Orion [A02] | V100 | 两个访存密集算子并发 | 只有小幅加速；计算 + 访存配对快 1.41× | §3、Table 2 |
| XSched [A05] | Jetson Orin | GPU 后台访存任务干扰 DLA 前台 | 前台 P99 为单独运行的 1.67×，统一优先级后 1.18× | §7.2、Fig.10 |
| Hummingbird [A09] | A100 | 空间共享下 LLM decode 受干扰 | Orion 下 TPOT 波动是独占的 10.9 倍；论文没有测带宽 | §3、Fig.5 |
| MuxWise [B13] | A100 / H100 | prefill 与 decode 并发 | v2：P90 < 7 %，最大 17 %（prefill 至少占一半 SM）；v3：0–30 %，称不可预测 | v2 Principle 1；v3 Fig.11 |
| Nexus [B14] | L20 | prefill 的 KV 变长干扰 decode | decode 最多慢 36 % | §3.3、Fig.6 |
| DuetServe [B16] | H100 | 带宽随 SM 数的变化 | 约 20 % 的 SM 就拿到约 60 % 峰值带宽 | §3、Fig.3a |
| Martín et al. [C20] | Orin Nano / AGX | 两个推理进程 | green context 不隔离内存；功耗限制会使隔离更差 | 结论节 |
| BWLOCK++ [C21] | TX2 | CPU 干扰 GPU | 最坏约 3× | Fig.7 |
| Serafini [C23] | Thor | GPU 干扰 CPU | 合并写 2.5×，跨步写约 4.5–5×；1–2 个 SM 就够 | Fig.6.10、6.12 |
| PR #193 [D24] | Thor | 7B 规划器干扰执行器 | 同进程加高优先级 stream 时慢约 16 %；没有带宽数据 | PR 正文 Validation 节 |

在统一内存 SoC 上，以下几种方向文献都没有量化：GPU 负载干扰 GPU 负载；CPU 负载干扰 GPU 上的 LLM；三方同时争用。

### 3.5 不可抢占单元的粒度

| 来源 | 单元 | 时长 |
|---|---|---|
| REEF [A01] | 小 DNN 的 kernel | 10–255 µs（VGG，Fig.10） |
| Hummingbird [A09] | LLM 推理中的 kernel；thread block | kernel 最长 7.49 ms；99.999 % 的 block 在 390 µs 以内（Fig.3） |
| LithOS [A06] | 训练 kernel | DLRM 超过 30 ms（§5） |
| Sarathi-Serve [B11] | 4k prompt 的一次迭代（Falcon-180B，流水并行） | 约 1150 ms（§3.3） |
| XSched [A05] | CUDA graph | 整个 graph 是一条命令（§9） |
| 本场景 | 规划器一个 2048 token 的 prefill 块（一个 graph） | 约 1.17 s；切更小会改变输出 |

### 3.6 前提对照

| 本场景前提 | 文献中具备该前提的工作 |
|---|---|
| P1 集成 GPU，CPU 与 GPU 共用 LPDDR 带宽 | A05（Orin 实验）、A07、C17–C23、D24、D25、D27 |
| P2 两个独立模型共置 | A 类全部、C17–C20、D24、D25 |
| P3 受保护方是周期性、截止期严格的小模型 | A07、C18；A01 和 C17 是固定频率但没有截止期；A05 的案例 2 是 25 FPS |
| P4 让步方是 LLM，且自己有时延要求 | B 类中 prefill 的 TTFT（同一模型）；A05 案例 2 用 laxity 策略保后台周期（whisper.cpp 每 3 s 一段）；A09 的 LLM 让步方没有时延要求 |
| P5 让步方的最小下发单元约 1 s，且切小会改变输出 | 只有 B11 的迭代时长同一量级（流水并行下）；文献中没有"切小会改变输出"这一约束 |
| P6 可用控制只有 MPS 百分比、green context、MPS 客户端优先级、stream 优先级 | B12、C20（MPS）；B13、B14、C20、C23（green context）；A02、D24（stream 优先级）；MPS 客户端优先级没有 |

同时具备 P1–P3 的只有 D24 和 D25，两者的规划器都是 7B，prefill 在百毫秒以内。D25 的全文没有拿到。

## 4. 本轮检索范围内没有找到的组合

1. 在集成 GPU 上调度两个独立模型的工作，其中一个模型有严格的周期截止期、另一个是 LLM。B 类专门做过一次检索，D 类只找到 PR #193 的测量。
2. 对 MPS 客户端优先级（`CUDA_MPS_CLIENT_PRIORITY`）的评估。
3. 统一内存 SoC 上 GPU 负载之间带宽干扰的量化，以及 CPU、GPU 三方同时争用的量化。
4. 不可抢占单元在秒级、且切小会改变输出时的调度。文献中的切分（thread block、层、token 块）都没有做输出一致性测试。
5. 机器人双系统里执行器的截止期错过率，以及执行器动作块余量与调度之间的关系。RTC [D29] 把推理时延写成执行器侧的 deadline d，但它的推理在远端。

## 5. 精读笔记与阅读顺序

| 顺序 | 笔记 | 读它回答什么 |
|---|---|---|
| 1 | [notes/2025-xsched.md](notes/2025-xsched.md) | 没有硬件抢占时，有哪些抢占层级、各需要什么；CUDA graph 为什么挡住抢占；Jetson Orin 上跨引擎的带宽干扰 |
| 2 | [notes/2026-hummingbird.md](notes/2026-hummingbird.md) | block 级切分怎么得到 µs 级抢占点；LLM decode 在空间共享下的干扰；切分依赖 PTX、不支持 cuBLAS 的限制 |
| 3 | [notes/2025-drift.md](notes/2025-drift.md) | 同一进程内用 green context 分 SM 的工程细节（预建分区、每个分区一套 graph、按层下发 prefill、按余量抢占）；v2 与 v3 的争用结论不同 |
| 4 | [notes/2026-serafini-thesis-ch6.md](notes/2026-serafini-thesis-ch6.md) | Thor 上带宽随 SM 数的曲线；限 SM 不等于限带宽；green context 粒度的文档说法 |
| 5 | [notes/2026-oxygen.md](notes/2026-oxygen.md) | Thor 上"约束动作频率、最大化另一方"的问题表述；MPS 并行在 Thor 上变慢；只有均值指标 |

## 6. 待核实

- **green context 在 Thor 上的 SM 粒度**：
  - 文档和 C23 都说 Tegra 是 2 SM 的倍数，C23 引的是 NVIDIA 文档、没有实测；CUDA Driver API 文档 v13.5 也这么写。
  - 本组实测请求 4 SM 实得 8 SM。可能的原因是分区 API 不同（`SplitByCount` 还是 `cuDevSmResourceSplit`）、对齐规则不同，或 CUDA 版本不同，都是推测。
  - 检查方法：在 Thor 上读 `minSmPartitionSize` / `smCoscheduledAlignment`，并用 `%smid` 测分区实际用到的 SM。
- **DualVLN on Thor [D25]**：OpenReview 全文需要人机验证，只读到摘要；两模型在 Thor 上的进程和 stream 结构未知。
- **会议元数据**：
  - MuxWise 的 DOI 未在 ACM DL 打开。
  - Bullet 的 ASPLOS '26 只由 camera-ready 首页确认。
  - FiS-VLA、RTC 的 NeurIPS 2025 没有在论文集页面核对。
  - 标 † 的条目见 [papers.md](papers.md)。
- **XSched 仓库 README 自述的功能**：支持 CUDA Graph、与 TSG-based Lv3 兼容，论文没有评测。
- **数值一致性**：Hummingbird 的 block 切分、MuxWise 的按层下发是否保持输出逐位一致，两篇都没有测。
