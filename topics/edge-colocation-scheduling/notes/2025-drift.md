# MuxWise（原名 Drift）：同一进程内用 green context 把 prefill / decode 放到不同 SM 上，decode 只拿"刚好满足 TBT"的 SM，prefill 按层下发以消除气泡并支持抢占

- **MuxWise（v1/v2 名为 Drift）** · 2026 · ASPLOS '26 · [链接](https://arxiv.org/abs/2504.14489)
- v3 标题：Towards High-Goodput LLM Serving with Prefill-decode Multiplexing；v1/v2 标题：Optimizing SLO-oriented LLM Serving with PD-Multiplexing
- 作者（v3）：Yukang Chen, Weihao Cui, Han Zhao, Ziyi Xu, Xiaoze Fan, Xusheng Chen, Yangjie Zhou, Shixuan Sun, Bingsheng He, Quan Chen（前三位 equal contribution）。v2 作者顺序为 Weihao Cui 第一，且包含 Minyi Guo、不含 Xiaoze Fan / Bingsheng He。
- 版本：arXiv v1 2025-04-20，v2 2025-04-22，v3 2026-02-07（arXiv 页面 Submission history 核对）。arXiv 页面无 Comments 字段；会议信息来自 v3 PDF 首页的 ACM Reference Format："Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '26), March 22–26, 2026, Pittsburgh"，DOI 10.1145/3779212.3790236（DOI 未在 ACM DL 打开核对 †）。
- 本笔记以 v3 为准；v2（Drift）与 v3 有实质差异的地方逐条标出；[papers.md](../papers.md) B13 卡片的数字来自 v2。

## 论文做了什么

在单个 serving 实例内，用 CUDA green context 在每块 GPU 上把 SM 分成 prefill 组和 decode 组，两阶段共享同一块显存与同一个 KV cache pool。decode 走 CUDA Graph，prefill 按 transformer 层切成若干段逐段下发，这样 prefill 可以在层边界改用新的分区、或被短请求抢占。调度器用 solo-run 延迟预测器乘以一个"争用最大放大系数"（contention guard）得到 decode 的最坏延迟，给 decode 分配满足 TBT SLO 的最少 SM，其余全给 prefill。基于 SGLang 实现，8×A100 / 8×H100 / 8×H200 上评估。

## 结构（v3）

- §1 Introduction：disaggregation 与 chunked prefill 的缺陷，提出 intra-GPU PD multiplexing。
- §2.1 LLM 服务与负载特征（Table 1，五类负载的输入 / 输出 / 复用长度）。
- §2.2 SLO 约束下 prefill / decode 的算力与显存需求随复用长度变化（Figure 3）。
- §2.3 现有方案缺陷：静态 / 动态分离（Figure 4、5），chunked prefill 的 token budget 两难（Figure 6）。
- §2.4 新范式与两个 challenge（C-1 气泡，C-2 未管理的争用）。
- §3.1 架构总览（Figure 8）。
- §3.2 Bubble-less multiplex engine：green context 选型、朴素集成的三类低效、按层执行 + 基于查询的同步。
- §3.3 Contention-tolerant estimator：争用测量（Figure 11）、solo-run 预测器（Eq. 1/2）、contention guard。
- §3.4 SLO-aware dispatcher：优先级、分区与按层数量、抢占（Figure 12）。
- §4 评估：真实 trace、SLO attainment / goodput、H100 / H200、合成负载、单卡、消融、开销。
- §5 Discussion：通用性、适用场景、大规模部署。
- §6 Related Work；§7 Conclusion；附录 Artifact。

## 问题表述

- 场景：数据中心多卡服务器。主测 8×A100-80GB（NVLink 600 GB/s），另有 8×H100-SXM5-80GB、8×H200-SXM5-141GB（§4.1）。模型 Llama-8B、Llama-70B（TP=8），Qwen3-235B（MoE，22B 激活）。负载：Mooncake 的 Conversation、Tool&Agent 真实 trace（缩放后，1 分钟内最高 13× 突发，Figure 13），以及 ShareGPT、LooGLE、OpenThoughts 合成负载（Poisson 到达）。无硬截止期，只有 P99 统计 SLO。
- 优化目标：goodput，定义为"peak throughput with SLO guarantees"（§1）。调度上"prioritize SLO attainment for the decode phase and process the prefill phase as early as possible"（§3.4.1），prefill 的 TTFT 不直接保证。
- 指标：P99 TBT（SLO：Llama-8B 50 ms，Llama-70B 100 ms，§4.1 Metrics），P99 TTFT，满足 99 %-ile TBT SLO 时的最大请求率；选 TBT 而非 TPOT，理由是 TPOT 是平均值会掩盖个别慢 token（§4.1）。
  - v2 差异：v2 给 Drift 设了 TTFT SLO（每 1K token 1 s，v2 §5.1 Metrics）；v3 删除了该设定，只在 §4.4.3 报告 "TTFT per token"。

## 核心观察

1. chunked prefill 的 token budget 无法同时满足 SLO 与打满 GPU。证据：Figure 6-(a)（8×A100，Llama3-70B，decode batch 32、每请求复用 1K），延迟直到 4K budget 才开始线性增长，此时 505 ms，远超 <100 ms TBT；§1 称 SLO 合规 budget 约 256（100 ms TBT），与打满所需 4K 相差 8×。
   - v2 差异：同一图在 v2 标注为 (4K, 240 ms)（v2 §2.3）。v3 为 505 ms，文中未解释变化原因。
2. prefill 中的复用上下文变长会让 chunked prefill 的 TBT 明显上升。证据：Figure 6-(b)，budget 固定 512，复用上下文超过 4K 后 TBT 明显上升（仅图，无具体数字）。
3. prefill 的算力需求随复用长度增长，decode 不敏感；两者 KV cache 可达数十至数百 GB。证据：Figure 3（Llama-70B，8×A100，所有 GPU 同样的部分 SM，找满足 SLO 的最小 GPU 比例；prefill batch 1、新 token 2K、TTFT 400 ms；decode batch 32、TBT 100 ms），仅图。
4. 分离部署缩小 KV pool 导致命中率下降，例如 36.6 % → 4.2 %（§1，Figure 5 为不同 pool 容量下的命中率；70B 达到最优命中率需 3.3 TB 显存，Figure 5 图注）。
5. 跨请求复用 KV 的代价对比（仅 v2 §1）：Llama-70B 上 1K 新 token + 127K 缓存 token 的请求，从 decode 实例迁移 >40 GB KV 需 133 ms，重算需 15.9 s（≥128× FLOPs），直接共享 KV 做 prefill 只需 205 ms。v3 删除了这组数字。
6. 空间复用下的争用（详见下方"专题 3"）：v3 的结论是争用 slowdown 在 ~0 到 ~30 % 之间、随分区配置变化大、"inherently unpredictable"（§3.3.1，Figure 11）；v2 的结论相反，是"multiplexing ... incurs minimal contention"（v2 §3.4）。

## challenge

- C-1：朴素集成产生 GPU 气泡（§2.4，§3.2.2）。三种：(a) prefill 下发耗时（数十 ms）长于一次 decode 迭代，下一次 decode 无法及时下发；(b) decode batch 全部结束时已下发的 prefill 仍按旧分区运行，GPU 不可抢占，释放的 SM 无法被利用；(c) 长 prefill 让短请求排队，SLO 余量小的短请求错过 TTFT。
- C-2：空间复用下内存带宽等共享资源不受管理，争用导致 SLO 违约（§2.4）；GPU 不暴露运行时带宽监控（§3.3.1）。

## 机制

### 1. 空间划分：green context（§3.2.1，v2 §4）

- 选型理由：MIG、MPS 是跨进程方案，"cannot provide flexible compute resource adjustment"，且 prefill 与 decode 之间要跨进程通信；green context 是进程内方案，"binding CUDA streams to specific SMs, with reconfiguration costing only a stream synchronization (on the order of microseconds)"，两阶段在同一进程，可共用一个 KV cache pool（v3 §3.2.1）。
- 同一组 green context 内各分区 SM 不重叠（v2 §3.4："There is no overlap among green contexts created within the same group"）。
- 依赖：CUDA Driver API green context；PyTorch 不原生支持，作者写了 PyTorch extension，使 PyTorch 下发的所有 GPU 操作都走 green context（v2 §4：约 10k 行 Python + 100 行 C++/CUDA）。要求驱动 570.124.06 / CUDA 12.8 "to ensure correct operation of GreenContext"（v2 §5.1）；附录要求驱动 > 570。
- 详见"专题 1"。

### 2. Bubble-less multiplex engine（§3.2.3）

- decode：整次迭代一个 CUDA Graph（下发 <0.5 ms），按若干 batch size 预先捕获。
- prefill：按 transformer 层切分（v3 称 prefill layers，PLs；v2 称 prefill blocks，PBs），用 piecewise CUDA graph（每层一个 graph），Llama-70B 8×A100 仍有约 10 ms 下发开销（v3 §3.2.2）。
- 每轮下发的层数 N_PL = ⌈(T_d × N_T) / T_P⌉，T_d 为预测 decode 延迟，T_P 为预测整个 prefill 延迟，N_T 为模型层数（§3.4.2）。即每次只下发"大约覆盖一次 decode 迭代时长"的那几层，CPU 能及时回来处理 decode 输出（解决气泡 a）。
- decode 结束后，后续 prefill 层切换到新的 green context（解决气泡 b）。
- 层边界即抢占点（解决 c，见"专题 4"）。
- Query-based synchronization：周期性轮询 CUDA event 而不是阻塞等待；最后一层 prefill 完成的事件一旦被看到，就把请求并入当前 decode batch（§3.2.3）。

### 3. Contention-tolerant estimator（§3.3）

- Solo-run 预测器：Eq. 1 T_prefill = θ1·Σn_i² + θ2·Σn_i·r_i + θ3·Σn_i + θ4；Eq. 2 T_decode = θ1·Σr_i + θ2·bs + θ3（n 新 token 数，r 复用 token 数）。v2 写明"For each compute partition configuration, a prediction model is trained"。最大误差 prefill 8.16 %、decode 8.84 %；离线 profiling 数小时，每个 LLM–机器组合一次（§3.3.2）。
- Contention guard（v3 新增）：对 5 个变量（prefill 新 / 复用 token 数，decode batch size，decode 总复用 token 数，分区配置）做网格采样，对每对 prefill / decode 返回所在网格单元观察到的最大 slowdown。只为 decode 提供系数。采样：token 维度按 4 的幂从 2K 到 128K，decode batch size 约 20 档，分区按 16 SM 粒度（A100 6 种、H100 7 种），共 (4×4−1)×4×20×6 ≈ 7K 个样本，约 12 小时；运行时用实际执行数据持续更新（§3.3.2）。
  - v2 差异：v2 称为 "contention-free modeling"，直接用 solo-run 预测器，不加争用系数（v2 §3.4）。

### 4. SLO-aware dispatcher（§3.4）

- 在每个 prefill batch 完成时和每次 decode 迭代结束时做决策（§3.4.2）。
- 给 decode 分配"best-fit"（满足 TBT SLO 的最少）SM，依据为 solo-run × guard 的最坏估计；其余给 prefill。prefill 的预测只需保证"下发的 prefill 层时长大于对应 decode 迭代"，不需要准确或最坏估计（§3.4.1）。
- 抢占：见"专题 4"。

## 专题 1：green context 的使用方式

| 问题 | 论文内容 | 位置 |
|---|---|---|
| 分区怎么建 | 服务启动前预建多组 green context，运行时按调度器决定切换；按需创建有运行时开销 | v2 §4 |
| 预建哪些分区 | v2：4 组 (prefill SM, decode SM) = (108,0)、(84,24)、(72,36)、(0,108)（A100 108 SM）。v3：按 16 SM 粒度，A100 6 种、H100 7 种配置 | v2 §4；v3 §3.3.2 |
| 分区数量的影响 | 3 组 (108,0)/(80,28)/(0,108) 控制 TBT 较差（decode 只有两档可选）；4 组与 5 组（多一个 (80,28)）效果相近 | v2 §5.3.2，Figure 13（P99 TBT 约 61–70 ms，仅图读数） |
| 切换代价 | "only a stream synchronization (on the order of microseconds)"；未给出实测数字 | v3 §3.2.1 |
| 内存代价 | 一组 green context 约 4 MB；与 CUDA Graph 联用时 v2 报 743 MB（Llama-8B 与 70B，8×A100），v3 报 6.2 % 开销（8×A100 / 8×H100），原因是每个 decode 分区都要为所有已记录的 batch size 再录一套 graph；"Due to the closed-source nature of CUDA, this cannot be addressed on our side" | v2 §5.3.3；v3 §4.5 |
| SM 粒度 | 论文没有给出驱动层面的最小 SM 粒度。v3 选 16 SM 的理由：(1) H100 及更新 GPU 上使用 thread block cluster 的 kernel "requires 16 SMs"；(2) 实验表明 16 SM 已够，更细收益小、显存开销更大。v2 的 A100 分区值 24/36/72/84 都是 12 的倍数，文中没有解释粒度来源 | v3 §3.3.2；v2 §4 |
| CUDA Graph 与 context 绑定 | 论文没有直接写"graph 绑定 context"这句话；它的处理方式等价于：decode 的 graph 按（decode 分区 × batch size）逐一捕获，因此内存随分区数线性增长，这也是 v2 只用 4 组、v3 用 16 SM 粒度的原因之一。另在 Related Work 中称 Bullet 的开源实现里每个 CUDA graph 的 SM 分配实际不会变化，并引用 libsmctrl 自己的说明"不支持 CUDA graph" | v2 §5.3.3；v3 §4.5、§6 |
| prefill 怎么跨分区 | prefill 不录整段 graph，按层下发（v3 用 piecewise graph），所以可以在层边界切到另一个 green context | v3 §3.2.2–3.2.3 |
| 可移植性 | 作者称 green context 自 Pascal 起支持，AMD 上可用 hipExtStreamCreateWithCUMask() | v3 §5（作者原话，未独立核实 †） |

## 专题 2："block-level" 切分切的是什么，是否改变数值

- 切的是模型深度方向的 transformer 层，不是 token。v2 §3.3："slicing the entire prefill phase into multiple prefill blocks (PBs). Importantly, block-wise slicing does not alter the internal execution model of the LLM, since LLMs are inherently composed of multiple transformer blocks." v3 §3.2.3 直接改名为层："MuxWise splits the prefill phase into layers (PLs) ... layer-wise execution incurs negligible overhead, since LLMs are inherently structured as multiple transformer layers." 每段内是整批 prompt 的全部 token 过若干完整层，层数由 N_PL 公式决定。
- 数值：论文只有上面这句定性论断，没有做输出一致性 / 数值误差实验。
- 判断：按层切分不改变任何 kernel 的输入形状，与 token 级 chunked prefill（改变 GEMM 的 M 维和 attention 的 query 段）机制不同，所以本场景"小于 2048 的 token 块会改变输出"的现象不会由层切分本身引入。但有一点论文未讨论：同一层在不同 SM 数的 green context 中运行时，若 GEMM / attention 库按 SM 数选择 tile 或 split-K 策略，结果可能逐位不同；需要实测。
- 开销：把 prefill 切到最细粒度（逐层）与整段下发相比，各种 batch size 与上下文长度下总开销 ≤1.5 %（v3 §4.5 Runtime；v2 §5.3.3 同）。消融：关掉按层下发，decode 的 P99 TBT（原文写作 "TTFT of decode"）约增加 10 ms；再关掉查询式同步，增加 314 ms（8B）/ 672 ms（70B）（v3 §4.4.2，Figure 19）。

## 专题 3：争用测量（v2 的 Principle 1 与 v3 的替换）

### v2（Drift）版本

- 论断："Principle 1. When there is no contention over compute units, only two GPU kernels that are both memory-bound will exhibit significant contention, leading to noticeable slowdown."（v2 §3.4，引用 [21, 34, 40, 41] 称其为 GPU 空间复用中广泛接受的原则）。
- 理论分析：A100-SXM4-80GB 按 320 TFLOPs / 2039 GB/s 算各 kernel 访存时间 / 计算时间比（v2 Table 2，batch 256）：QKV 0.666、O 0.765、UG 0.628、D 0.659、Extend Attn（新 1024 / 复用 8196）0.027、Decode Attn（复用 1024）19.2。只有 decode attention > 1，结论是 prefill 侧全部 compute-bound，因此两侧"都 memory-bound"的情形不出现，争用很小。
- 实测：Llama-8B / 70B，8×A100；prefill batch 总上下文 1024–32,768 token；decode 的 batch size × 每请求复用长度 1024–4,194,304；prefill 至少拿一半 SM（"we find this rarely happens in online serving" 指的是 prefill 少于一半 SM 的情形）。8600 多个组合，两阶段的争用 slowdown P90 <7 %，最大 17 %。据此直接用 solo-run 预测器，不建模争用（v2 §3.4）。
- 两侧都 memory-bound 时会怎样：v2 只在原则层面说"会有显著争用"，没有给出这种情形下的测量数据，理由是 Table 2 显示现代 LLM 上 prefill 不会 memory-bound。

### v3（MuxWise）版本

- 删除了 Principle 1 和 Table 2 的访存 / 计算比论证，改为："both prefill and decode phases can heavily consume bandwidth, making contention hard to predict"，且 GPU 不暴露运行时带宽监控（§3.3.1）。
- 测量：Llama-8B / 70B，8×A100 与 8×H100；横轴为 decode 分到的 SM 数（A100 图上刻度 12–76，H100 20–100），其余给 prefill；prefill 总上下文 1,024–128K，decode 复用长度 1,024–1,024K；"This profiling takes one week"（§3.3.1，Figure 11）。
- 结果：decode slowdown 从接近 0 到约 30 %，随分区配置变化大，作者称"inherent unpredictability"；两模型在同一 GPU 上趋势相似；prefill 的 slowdown "similar but omitted"（§3.3.1）。contention guard 一节给出上界：A100 最大 20 %，H100 最大 30 %（§3.3.2）。
- 判断：v3 覆盖了 decode 拿多数 SM（prefill 少于一半）的区间，也把上下文扩到 128K / 1M，所以最大值从 17 % 升到 20–30 %。论文仍没有给出 slowdown 的分布（只有每个配置的范围，仅图），也没有按"两侧都 memory-bound"分组报告。
- "8600 多个组合、P90 <7 %、最大 17 %、prefill 至少一半 SM"只出现在 v2，v3 已替换；引用时写明版本。

## 专题 4：抢占算法（v2 Algorithm 1；v3 Figure 12 文字描述）

- 输入（v2 Algorithm 1）：请求队列 Q_req；当前 prefill batch PB；被抢占 batch 的栈 Stack_PB；当前 decode batch DB；两侧分区 C_PB、C_DB；TBT SLO；延迟预测 predicting(batch, DB, C_PB, C_DB)（solo-run 模型，v2 称 contention-aware model）；被抢占 batch 的 SLO 余量 PB.SLO_headroom()。
- 流程（GeneratePB，第 6–20 行）：
  - 栈非空：若当前 PB 为空，弹出栈顶恢复；否则继续当前 PB（此时不再考虑新 batch）。
  - 栈空：取新 batch PB_new；若无当前 PB，直接执行 PB_new。否则预测 T_PB 和 T_PBnew，若 T_PB + T_PBnew ≤ PB.SLO_headroom()，把 PB 压栈、执行 PB_new；否则把 PB_new 放回队列，继续 PB。
  - 判断：条件只检查被抢占方 PB 能否仍在其 TTFT 余量内完成，不检查 PB_new 自身的 SLO；T_PB 是 PB 的总预测时间而非剩余时间（伪代码原样如此，文中未说明）。
- 深度：栈大小为 1，一个 batch 至多被抢占一次，且抢占者不能再被抢占（v2 §3.5："preemption stack size of one"；v3 §3.4.2："MuxWise does not allow recursive preemption"）。理由：通常是短请求抢长请求，再抢短请求会让它违约。只在发生抢占时检查 SLO，其余时间只求尽快完成当前 prefill。
- 物理上是什么：不是中断 GPU kernel（论文明确说 GPU 执行不可抢占，v3 §3.2.2）。抢占发生在层边界：已下发的层跑完，之后不再下发被抢占 batch 的后续层，转而下发新 batch 的层；被抢占 batch 之后从栈中恢复，继续剩余的层。判断：被抢占 batch 已算完的层的中间激活与 KV 需保留在显存中，论文没有说明其存放与开销。
- 分区随之调整：Partition（第 21–24 行）对新的 PB 重新选 C_PB / C_DB 并生成本轮要下发的层。
- v3 说明抢占是可选的，关闭时 MuxWise 仍明显优于基线（§3.4.2）。
- 评估：ShareGPT 与 LooGLE 各 50 % 混合，0.5 req/s Poisson；开启抢占使 P99 "TTFT per token" 快 1.96×（v3 §4.4.3，Figure 20）。

## 评估

- 基线（v3 §4.1）：SGLang chunked prefill（按 Sarathi-Serve 方法离线调 token budget，FlashInfer 融合 prefill / decode attention，作者称性能应与 POD-Attention 相近）；NanoFlow（与 chunked 同 budget）；LoongServe（70B：SP2×TP4；8B：SP4×TP2）；SGLang-PD（静态分离，P:D = 1:1，各 TP4）。Dynamo 测过但明显差于 SGLang-PD，未列入。
  - v2 基线：SGLang chunking（budget 512）、LoongServe、Dynamo（vLLM 后端，8B P:D=2:1，70B 1:1）。
- 负载与平台：见"问题表述"。v3 使用 SGLang 0.4.10post2、PyTorch 2.6.0、驱动 570.124.06、CUDA 12.8（v2：SGLang 0.3.6post3、PyTorch 2.6.1）。
- 主要结果（v3）：
  - 真实 trace：P99 TTFT 平均快 3.57×（chunked）、5.98×（NanoFlow）、4.65×（LoongServe）、1.66×（SGLang-PD）；MuxWise 与两个分离方案都满足 TBT SLO，chunked 与 NanoFlow 多数情况不满足；SGLang-PD 的 TBT 比 MuxWise 更短（§4.2.1，Figure 14）。
  - Goodput（Tool&Agent，Poisson）：8B 对四个基线 2.6×、5.2×、2.0×、1.3×；70B 对 chunked、LoongServe、SGLang-PD 3.06×、2.62×、1.62×（NanoFlow 在 70B 上 chunk 64 也不满足 SLO）；TTFT 最多快 9.16×（§4.2.3，Figure 15）。Table 5：8B 下 MuxWise 7430 token/s、GPU util 84.0 %；70B 下 25397 token/s、88.1 %。
  - H100（8B/70B）与 H200（Qwen-235B），只和 chunked 比：P99 TTFT 平均 2.28×，P99 TBT 平均 1.81×（§4.2.4，Figure 16）。
  - 合成负载（70B）：ShareGPT 1.9× / 1.73× / 9.5× / 1.46×；LooGLE 1.71× / 2× / 1.33× / 2×；OpenThoughts 对 chunked、NanoFlow、SGLang-PD 均 2×，LoongServe 始终不满足 SLO（§4.3，Figure 17）。
  - 单卡：Llama-8B 单 A100、ShareGPT，goodput 比 chunked 高 1.2×，TBT 相近（§4.3.1）。
  - 摘要：平均 2.20×（最多 3.06×）goodput。
  - v2 摘要：平均 5.1×（最多 17.5×）吞吐；真实 trace 上 P99 TTFT 平均 2.62× / 7.72× / 9.88×（chunking / LoongServe / Dynamo），TBT 2.59× / 2.22× / 2.67×；goodput 8B 6.2× / 1.63× / 2.10×，70B 17.5× / 1.4× / 1.75×（v2 §5.2）。
- 消融：
  - 气泡消除：见专题 2（Figure 19）。bubble ratio（Nsight Systems，流上无 kernel 的时间占比，两条流取平均）MuxWise 7.7 % vs chunked 4.5 %，额外气泡出现在 prefill 全部完成、只剩 decode 时（§4.4.2）。
  - 分区随负载变化：LooGLE 多数 SM 给 prefill，OpenThoughts 多数给 decode；突发区间内 30 s 内用到了全部 6 种配置（§4.4.1，Figure 18）。
  - 分区组数：见专题 1（v2 §5.3.2）。
  - 抢占：1.96×（§4.4.3）。
  - 相关工作中的对照：WindServe 用普通 CUDA stream 复用，作者原型复现下 MuxWise 在 A100 / Llama-8B / ShareGPT / 50 ms TBT 上 goodput 高 1.61×；只做时间复用（按层切分以填小空隙）的增强版 Tropical 至少差 20 %（§6）。

## 作者自述的局限

- prefill 的 TTFT 不直接保证；作者认为 prefill 违约说明负载已超过实例峰值容量（§3.4.1）。
- 只做单实例内调度（§3.4.1）。
- SLO 宽松或无 SLO（离线服务）时没有机会胜过 chunked prefill 或 NanoFlow（§5）。
- prefill 实例持续处理长请求或复用 decode 会违反 TTFT 时，在分离部署中收益有限（§5）。
- 争用无法管理，只能用离线最坏系数兜底；GPU 不暴露带宽监控（§3.3.1）。
- CUDA Graph 与 green context 联用的显存开销因 CUDA 闭源无法在作者侧解决（v2 §5.3.3）。

## 与本场景的前提对照

| 前提 | 论文 | 本场景（见 [survey.md](../survey.md) §0.2） | 是否成立 |
|---|---|---|---|
| 共置对象 | 同一模型的 prefill 与 decode，同一进程 | 两个不同模型（执行器 ImageWAM + 规划器 Qwen） | 不成立；green context 是进程内机制，两模型需在同一进程中才能照搬 |
| 硬件 | 8×A100 / H100 / H200，HBM 2–4.8 TB/s 级，108/132 SM，TP=8 | 1 块 Thor，20 SM，LPDDR5X 约 273 GB/s，CPU 共享带宽 | 不成立 |
| 分区粒度 | 16 SM（v3）；共 6–7 种配置 | green context 8 SM 粒度，20 SM 只有极少几种切法 | 部分成立；可选配置远少于论文 |
| 争用大小 | 最大 20 %（A100）/ 30 %（H100），只为 decode 建 guard | decode 带宽占用是按 SM 划分后的主要剩余干扰；CPU 内存负载让 decode 慢约 2.2 倍 | 不成立；数量级不同，且本场景受害方是执行器而非 decode |
| SLO 形式 | P99 TBT 50/100 ms，P99 TTFT，统计 | 执行器周期硬截止期 200 / 533 ms | 不成立 |
| 受保护方 | decode（迭代短、可用 CUDA Graph） | 执行器（37.5 ms 单次，周期调用） | 结构相似（判断）："给受保护方刚好够的 SM、其余给 prefill" |
| prefill 切分方式 | 按层切，不改 kernel 形状，开销 ≤1.5 % | 按 2048 token 块 forward，更小 token 块会改变输出 | 层切分可作为不改 token 块的另一种可抢占粒度（判断，数值需实测） |
| 抢占 | 层边界停止下发，栈深 1，按 TTFT 余量判断 | 规划器 prefill 1.2–2.2 s 期间执行器需按周期插入 | 机制可借鉴；判断依据要换成执行器截止期 |
| 分区与 CUDA Graph | 每个分区一套 graph，显存线性增长 | 执行器若用 graph 也需每分区一套 | 成立（判断） |

## 可以借用的 / 不能借用的

- 可借用（判断）：
  - 规划器 prefill 按层下发、每次只下发约一个执行器周期能覆盖的层数，使规划器在层边界让出或换分区；比缩小 token 块更不容易改变数值。
  - 预建少量分区 + 运行时只切换，不在线创建 green context。
  - 用 CUDA event 轮询代替阻塞同步，避免一侧等待另一侧完成。
  - "solo-run 预测 × 离线测得的最大争用系数"作为最坏延迟估计，给受保护方选最少 SM；本场景需把 CPU 内存负载也作为网格维度。
- 不能借用（判断）：
  - v2 的"prefill compute-bound 故争用可忽略"结论：带宽低一个数量级的 LPDDR5X 上已观测到相反现象。
  - 统计 P99 目标与吞吐导向的 goodput 定义；本场景要的是每个周期不违约。
  - 进程内 green context 前提：两个模型若在不同进程，green context 不能跨进程共享 SM 划分，需 MPS 或合并进程。

## 待核实

- ASPLOS '26 收录与 DOI 只在 v3 PDF 首页核对，未在 ACM DL / DBLP 上打开。
- 切换代价"微秒级"无实测数字；green context 的驱动级最小 SM 粒度论文未给出（本场景的 8 SM 粒度来自本组的 Thor 实测，不来自本文）。
- 按层切分的数值一致性论文未测；不同 SM 数分区下 GEMM 算法选择是否变化未讨论。
- 743 MB（v2）是 8 卡合计还是每卡，原文表述不明；v3 改为 6.2 %，基数（占显存还是占什么）未说明。
- v2 → v3 中 Figure 6-(a) 的饱和点延迟从 240 ms 变为 505 ms，原因未说明。
- Figure 11 仅图，没有分布统计（分位数）；"两侧都 memory-bound"情形无单独数据。
