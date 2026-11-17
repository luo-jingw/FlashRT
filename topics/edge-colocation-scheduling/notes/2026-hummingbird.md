# Hummingbird：把低优先级 kernel 按 thread block 切成一波（one wave）大小的 split-kernel，在 NVIDIA 闭源 GPU 上得到 µs 级抢占，并只在高优先级的空闲气泡里运行低优先级

- **Hummingbird** · 2026 · arXiv 预印本 · [链接](https://arxiv.org/abs/2601.04071)
- Hummingbird: SLO-Oriented GPU Preemption at Microsecond-scale
- 作者：Tiancheng Hu, Chenxi Wang\*, Ting Cao, Jin Qin, Lei Chen, Xinyu Xiao, Junhao Hu, Hongliang Tian, Shoumeng Yan, Huimin Cui, Quan Chen, Tao Xie\*（北大、国科大、清华 AIR、华科、蚂蚁、上交；\*通讯作者）；arXiv v1 2026-01-07，v2 2026-02-10（本笔记读 v2，cs.DC）。
- 元数据核对：arXiv abs 页面（标题、作者、版本日期）。abs 页面**没有 Comments 字段**，PDF 首页与页脚也没有会议印记，因此标"预印本"，会议未知。

## 论文做了什么

在一块 NVIDIA GPU 上同时运行一个高优先级（HP）在线 LLM 服务和一个低优先级（LP）离线推理或训练任务。系统拦截 CUDA Driver API，在运行时改写 LP kernel 的 PTX，让一个 kernel 可以按 thread block 子集分多次 launch（split-kernel）。HP kernel 到达时，调度器停止下发 LP 分段，等待时间不超过一个分段的执行时间（平均 139 µs）。LP 只在 HP 的"气泡"中运行：小气泡用 host 侧 API 模式（cudaMemcpyAsync + cudaStreamSynchronize、NCCL 调用）识别，大气泡用超时识别；大气泡期间把分段合并回原始 grid。另外有一个基于 NVLink 的分层显存卸载模块。

## 结构

- §1 Introduction：动机数字（kernel 时长分布、block 时长、小气泡占比）与结果概要。
- §2 Background and Observations：§2.1 kernel/block 时长的多样性与可预测性；§2.2 NVIDIA 硬件限制（不能抢占已提交 kernel）。
- §3 Motivation：在 A100 上用 llama.cpp 测 Orion（空间共享）与 REEF（时间共享）的 SLO 达成率、TPOT 波动、LP 吞吐（Fig.4、Fig.5）。
- §4 Design：§4.1 概览（Fig.6）；§4.2 Kernel Splitter（最优切分大小、PTX 改写）；§4.3 Runtime Scheduler（Algorithm 1、气泡检测、consolidation、kernel-tick）；§4.4 Memory Management（NVLink 扩展的 unified memory）。
- §5 Implementation：CUDA API hook、profiler、基于 NEUTRINO 的 PTX 改写、闭源库与 CUDA graph 的处理。
- §6 Evaluation：6.1 单 GPU（Fig.10）；6.2 显存紧张场景（Fig.11）；6.3 分解（Fig.12）、抢占延迟（Fig.13a）、多 GPU（Fig.13b）、跨 GPU 型号（Fig.14）。
- §7 Related Works；§8 Conclusion。
- 补充材料：§A.1 6 个模型 × 6 个框架的小气泡模式（Table 2、Table 3）；§A.2 从 Nsight Systems trace 自动发现气泡模式；§B 基于 HUVM 驱动的显存管理实现。

## 问题表述

- **场景**：数据中心 GPU。单 GPU 主实验为 8×A100-80GB SXM4 服务器中的一块，CUDA 12.6，关闭 DVFS（§6.1 Testbed）。另测 L40s、H100（Fig.14），以及 16×A100（两台 AWS p4de.24xlarge，NVLink 3.0 600 GB/s，节点间 400 Gbps）（§6.3）。HP：llama.cpp serving 模式的 Llama-8B 或 Yi-34B，8-bit 量化，BurstGPT 真实 trace；LP：batch 32 的 Mistral-7B / DeepseekMoE-16B 离线推理（llama.cpp），以及 PyTorch 的 ResNet-101 / GPT-2 训练（§6.1 Applications，Table 1）。
- **SLO**：HP 独占运行时测得的 P99 TTFT 与 P99 TPOT（§6.1 Metrics）。没有周期性截止期；LP 没有任何时延或吞吐要求。
- **优化目标**："how to maximize GPU utilization at the premise of ensuring SLO goals?"（§1）。设计原则："high-priority tasks execute with strict performance isolation, while low-priority tasks opportunistically harvest idle time slices"（§1 Major insights）。
- **指标**：HP SLO 达成率（满足 SLO 的请求比例）；LP 吞吐（tokens/s，或相对 LP 独占归一化）；GPU 利用率（Nsight Systems 的 SM active ratio）；平均抢占延迟（µs）。

## 核心观察

1. **kernel 时长跨三个数量级，等待式抢占的延迟不可预测。** 证据：§1、Fig.3a。A100 上 Llama-8B 推理的 kernel 从 5 µs（elementwise）到 7.49 ms（GEMM）；§2.1 称最长可达"tens of milliseconds"。
2. **单个 thread block 的时长短且稳定。** 证据：§1、§2.1、Fig.3b。CNN/LLM/MLLM 的推理、微调、训练中 99.999% 的 block 在 390 µs 内完成（§1；§2.1 写作"within 400µs"）。Fig.3b 图例给出各负载最长 block（仅图）：MLLM_Infer 212、LLM_Train 390、CNN_Train 80、CNN_Infer 87、LLM_Finetune 316、LLM_Infer 369 µs。可预测性的论据是 DNN kernel 的 block 很少有条件分支和可变循环，且负载是迭代式的；llama.cpp、SGLang 改变 prompt 长度时只改变 kernel 数或 block 数，不改变单个 block 时长（§2.1 Predictability）。
3. **HP LLM 推理中存在大量 µs–ms 级小气泡。** 证据：§1、Fig.2a（数值为图中标注）。一次请求中小气泡占 GPU 时间：Llama-8B 15.8%，DeepSeek-16B 17.2%，Llama-70B（TP）24.9%，GPT-oss-120B（EP）32.4%；分布式比单卡高 1.8×。来源：每次迭代的 D2H token 拷贝与同步、continuous batching 的元数据更新、跨 GPU 通信、CPU 侧 launch 不足。补充 Table 2/3 给出各模式时长：streaming 的 cudaMemcpyAsync+cudaStreamSynchronize 500–1000 µs，continuous batching 700–1500 µs，NCCL 类 150 µs–20 ms。大气泡（请求间隙）在 GPT serving trace 中最多占 23.6% GPU 时间（§1）。
4. **空间共享（Orion）在 LLM decode 下有严重带宽干扰。** 证据：§3、Fig.5（问题 3 详述）。
5. **时间共享（REEF）受长 kernel 与 eviction 开销限制。** 证据：§3、Fig.5a。REEF 的 SLO 达成率比独占差 1.5×；LP 吞吐比 Orion 低 2.1×；23.4% 的 LP 计算因 eviction 与 relaunch 被浪费；device queue 限 4 个 kernel 带来的同步开销占 6.8% 的 GPU 周期；一个 LP kernel 最多被重新 launch 3 次。
6. **概览数字。** 证据：Fig.1，坐标为（GPU 利用率，SLO 达成率），仅图：Exclusive (33%, 99%)、Hummingbird (82%, 98.4%)、REEF (67%, 72.3%)、Multi-streams (99%, 7.2%)、MPS (98%, 2.5%)、LithOS (92%, 31%)、Orion (95%, 12.5%)、MIG (87%, 0.1%)。图中没有说明对应哪一组负载。

## challenge

- NVIDIA GPU 不支持抢占已提交的 kernel；HP 必须等正在运行的 LP kernel 结束（§1、§2.2）。
- MPS、stream 优先级、LithOS、Green Context 只能控制计算资源，L2、HBM 带宽、PCIe 带宽不受控（§1）；MIG 不能动态重新划分（§1，§7 引用 MIG 重配置需数百 ms）。
- 切分粒度的权衡：分段越小，抢占越快、越能填进小气泡，但会降低利用率并增加 launch 开销（§1 第一个 challenge）。
- LP 被频繁抢占时的同步与 relaunch 开销（§1、§3）。
- 没有在线的 µs 级 GPU 空闲监视器，无法直接检测气泡（§4.3）。
- 模型变大后显存容量成为共置瓶颈（§4.4）。

## 机制

### 1. Kernel Splitter（§4.2、§5）

- **最优切分大小**：第一步按 SM 容量取一波，`N_block = N_SM · o · SM_MAX_THREADS / THREADS_PER_BLOCK`（式 1，o 为 occupancy）。第二步从 N_block 开始逐步减少 block 数并测执行时间；若时间变短，说明 kernel 受带宽限制，继续减少直到时间稳定（§4.2）。因此**分段大小以"刚好填满 SM 或刚好打满带宽"为准，不以 400 µs 为目标**；400 µs 是"一波 block 的时长上界"这一观察给出的抢占延迟上界。平均分段时长 77 µs（§1）。
- **离线/在线 profile**：用 cudaEvent 测 kernel 时长，用 CUDA API 查硬件参数与 occupancy；每个任务约需数秒（§5 Kernel profiler）。

### 2. PTX 改写（§4.2、§5）——问题 1

- **机制**：hook `cuModuleLoad` / `cuModuleGetFunction`，用 NEUTRINO 的 probe engine 对加载的 ELF/FatBinary 做 objdump 取出 PTX，按 kernel 名裁成单 kernel 汇编；在参数列表末尾加 offset_x/y/z 三个参数，插入 `ld.param` 读取，再在读取 `%ctaid` 后插入 `add` 指令加上偏移，"thereby preserving the original addressing semantics"；最后用 ptxas 汇编成机器码。每个 kernel 只改写一次（§4.2 Listing 1、§5 Kernel transformation）。
- **需要什么**：不需要 kernel 源码，也不需要重新编译应用；但需要二进制中**带有 PTX**，并在运行时用 ptxas 重新汇编。选 PTX 层是为了同时覆盖 AOT 的 CUDA 库和 JIT 的 Triton（§4.2）。作者称已改写 CUTLASS 和 Triton 中的复杂 kernel，并集成进 PyTorch（§4.2）。
- **闭源库**："A limitation is that we can not acquire the PTX of closed-source libraries such as cuBLAS and cuDNN"；做法是把其中大部分换成 CUTLASS（§5）。LithOS 用逆向工程切分闭源 kernel，作者认为开销更大（大量提前返回的短线程），并称 LithOS 开源后可作为补充（§5）。
- **不能切的 kernel**：使用跨 block 同步（`grid_group::sync()`）的 kernel 和 persistent kernel，直接关闭切分；作者称可通过源码重构解决（§5 Generality of kernel splitting）。
- **CUDA graph**："Hummingbird **can** intercept graph creation APIs and split graphs into subgraphs, ensuring correct execution ordering, similar to previous work [LithOS]. As long as we keep the subgraphs' execution no longer than 400µs, we can still provide a preemption latency guarantee"（§5）。措辞是"can"，评估中没有使用 CUDA graph 的负载（llama.cpp serving 与 PyTorch eager 训练），因此 graph 支持未经实验验证。
- **数值结果**：论文只说偏移保持原有寻址语义（§4.2），**没有讨论或测量数值一致性**。判断：每个 block 看到的 blockIdx 与原 kernel 相同，若 kernel 只依赖 blockIdx 且 block 之间不通过 atomic 累加交互，则每个 block 计算的内容相同、结果逐位一致；以下情况论文没有说明，可能不一致或出错：(a) kernel 读取 `%nctaid`（gridDim）——论文只描述了平移 `%ctaid`，没有描述修正 `%nctaid`，分段 launch 时 gridDim 会变小；(b) 依赖 block 间 atomic 的规约（如 split-K / stream-K GEMM），分段后执行顺序改变，浮点累加顺序可能变化；(c) 依赖"最后一个 block"计数器的 kernel。

### 3. Runtime Scheduler（§4.3，Algorithm 1）

- **HP 优先**：HP 队列非空时立刻停止下发新的 LP 分段并 launch HP kernel（Lines 3–5）。平均抢占延迟 139 µs（§4.3.1）。
- **气泡检测**：小气泡用 host 侧提示——在特定 API 模式（如 cudaMemcpyAsync 后接 cudaStreamSynchronize、ncclSend/Recv）前后插入 cudaEvent 标记；补充 §A.2 用 Nsight Systems trace 自动发现模式，并在与训练任务共置时验证，若抢占延迟退化超过 1% 则剔除该模式。大气泡：HP 超过阈值时间无 kernel 即判定，阈值略高于该任务观测到的小气泡时长（§4.3）。
- **Consolidation**：大气泡期间把分段合并回原 grid；用请求间隔预测器（N-BEATS）预测下一间隔，保证合并后的执行时间不超出间隔（§4.3）。
- **kernel-tick**：LP 运行时 device queue 中最多 1 个 LP kernel，从而抢占延迟 ≤ 一个分段时长。为避免每个 kernel 后同步（同步 API 约 5 µs，再加 6–7 µs launch 延迟），按预测的 kernel 时长减去 launch 开销计算下发间隔，在当前分段快结束时下发下一个（§4.3）。多个 LP 任务按 round-robin 服务（§4.3）。
- **依赖**：只依赖拦截 CUDA Driver API（`cuLaunchKernel`、`cuMemAlloc` 等），不修改应用（§5 CUDA API hook）；"requires no hardware-specific instructions and supports any generation of GPU architectures"（§5）。不依赖 MPS、green context 或未公开 ioctl。HP 与 LP 需经过同一个调度器（判断：拦截层要看到两边的 launch 才能判断 HP 队列是否为空；论文没有说明 HP 与 LP 是否在同一进程/同一 context）。

### 4. Memory Management（§4.4、补充 §B）

- 基于开源 HUVM（修改过的 NVIDIA UVM 驱动），2 MB chunk 粒度；HP 的内存用 `cuMemAdvise(SET_PREFERRED_LOCATION)` 钉在本卡，只驱逐 LP 页；驱逐目标按 NVLink 拥塞分数选择（4 MB 探测包测时延，Tcurr/Tbase > 1.5 视为拥塞，探测周期 10–100 ms），NVLink 不可用时退回主机 DRAM。依赖：修改驱动、NVLink 多卡。

### 实现规模

约 8000 行 C++/CUDA（§5）。

## 评估

- **基线**：REEF（device queue 容量 4，即 REEF-N 等待式）、Orion（HP 放最高优先级 stream，LP 放默认 stream）、LithOS（作者自行复现：libsmctrl 动态 TPC 映射、开源 kernel 上的 atomization、TPC stealing，超参取 LithOS 论文值）；Fig.1 另有 MPS、Multi-streams、MIG；独占（§6.1 Baselines）。
- **负载**：Table 1。HP：Llama-8B、Yi-34B（serving）；LP：Mistral-7B、DeepseekMoE-16B（batch 32 推理）、ResNet-101（batch 64 训练）、GPT-2 124M（batch 16 训练）。Table 1 中独占时平均 SM active / 带宽利用率：Llama-8B 55.8% / 38.7%，Yi-34B 58.3% / 53.0%，Mistral-7B 63.4% / 46.5%，DeepseekMoE-16B 75.1% / 55.0%，ResNet-101 88.8% / 16.4%，GPT-2 91.2% / 18.4%。
- **主要结果**：
  - 单 GPU（§6.1，Fig.10，柱高仅图）：Orion 所有组合 SLO 达成率 ≤ 22.8%，LP 为训练时 ≤ 10%；LithOS 比 Orion 高 1.8×；REEF 比 LithOS/Orion 高 2.7×，但 LP 为 GPT-2 训练时降到 14.9% 以下（长 kernel）；Hummingbird 所有组合接近 99%，平均比 LithOS 高 5.6×、比 REEF 高 3.0×；最大抢占延迟上限 400 µs，HP 变慢 < 1%。LP 吞吐平均为 REEF 的 1.9×，HP 为 Yi-34B 时 2.5×；GPU 利用率 Hummingbird 82%、REEF 67%。
  - 显存紧张（§6.2，Fig.11）：HP Llama-8B / Yi-34B + LP Llama-70B 推理，LP 需换出约 13% / 47% 显存；Orion、LithOS 的 HP SLO 达成率最高仅 12%；Hummingbird 比 REEF 高 5.6×，相对独占下降 < 3%；LP 吞吐为 REEF 的 4.2×。
  - 分解（§6.3，Fig.12，Llama-8B + Mistral-7B）：HP P99 TPOT 独占 10.6 ms，REEF 12.5 ms（正文；图中标注为 13.5 与 12.4，对应关系不明），Hummingbird 10.9 ms。LP 吞吐（图中标注，tokens/s）：REEF 66.8 → 仅切分 42.2（正文：切分使吞吐下降 37%）→ +consolidation 60.3（正文称 1.5×，按标注为 1.43×）→ +kernel-tick 86.0（正文 1.43×）；相对 REEF 1.29×；相对不切分、不同步的 vanilla 实现只慢 1.3%。consolidation 对总体抢占延迟的影响 < 0.7%。
  - 抢占延迟（§6.3，Fig.13a，x 轴为 MIST、LMA、Yi、GPT2）：平均 121–165 µs，比 REEF 快 4.3–6.6×。
  - 多 GPU（§6.3，Fig.13b）：HP Llama-405B（节点内 TP=8、节点间 PP），LP GPT-2 DDP 训练；SLO 达成率为 REEF 的 9.7×，LP 吞吐为 REEF 的 3.3×；分布式下小气泡最多占 30%。
  - 跨 GPU（§6.3，Fig.14，Llama-8B + Mistral-7B）：L40s / A100 / H100 上 P99 TPOT 分别比 Orion、REEF 低 39.0%、31%、13.3%（原文如此，三个数与两个基线的对应关系表述不清），接近独占；LP 吞吐为 REEF 的 1.25×。
- **消融**：即 Fig.12 的 splitting / consolidation / kernel-tick 三步。没有对切分大小（例如半波、两波）做扫描。

## 问题 2：低优先级为 LLM 推理的配置

- **有**：LP 为 Mistral-7B 与 DeepseekMoE-16B 的 batch 32 离线推理（§6.1），以及 Llama-70B 推理（§6.2）。全部用 llama.cpp 实现，论文**不区分 LP 的 prefill 与 decode**，也没有给出 LP 的 TTFT/TPOT 或单次 forward 时延。
- **抢占延迟**：Fig.13a 中 MIST 一栏为 LP 是 Mistral-7B 的配置（判断：x 轴同时出现 MIST、LMA、Yi、GPT2，图注未说明每栏是 HP 还是 LP）；全部配置平均 121–165 µs（§6.3），§4.3.1 给出均值 139 µs。没有单独给出 LP 为 LLM 时的数值。
- **LP 吞吐损失**：
  - 相对 LP 独占的归一化吞吐只在 Fig.10b 中（仅图），正文没有给数。
  - Llama-8B + Mistral-7B：Hummingbird 下 LP 86.0 tokens/s，REEF 66.8 tokens/s（Fig.12b 标注）；相对"不切分、不同步"的 vanilla 共置实现慢 1.3%（§6.3）。论文没有给出 Mistral-7B 独占吞吐，因此无法从正文算出相对独占的损失。
  - 仅切分不做 consolidation 与 kernel-tick 时，LP 吞吐下降 37%（§6.3）——这是切分本身的代价（一波一个 launch + 每段同步）。
  - Llama-70B 作为 LP：吞吐为 REEF 的 4.2×（§6.2），绝对值仅图。
- **LP 的时间来源**：LP 只在 HP 气泡中运行（Algorithm 1 Lines 6–9），HP 有 kernel 待发时 LP 完全停下。论文中 HP 是 serving 负载，独占时 GPU 利用率仅 33%（Fig.1），LP 主要吃大气泡（请求间隙）与 15–32% 的小气泡。
- **与本场景反向设置的关系**（判断）：本场景 HP 为执行器（37.5 ms/200 ms，占约 19% 时间），LP 为 27B 规划器。按 Hummingbird 的策略，规划器只能在执行器空闲的约 81% 时间内独占 GPU，与执行器完全不重叠，因此执行器不受规划器的带宽干扰；规划器的损失约为执行器占用比例加分段开销。执行器内部若有小气泡，规划器的分段会被插入，执行器每次恢复最多等一个分段。论文没有评估"LP 是大模型、HP 是小而密集的周期任务"的配置。

## 问题 3：Orion 的带宽干扰分析

- **设置**（§3 Setup）：A100；HP 为 llama.cpp serving 的 Llama-8B（8-bit，BurstGPT trace），LP 为 Mistral-7B 对 ShareGPT 对话做摘要（batch 32，8-bit）；Orion、REEF 参数取原论文值；SLO 为 HP 独占 P99。
- **数字**（§3、Fig.5）：
  - Fig.5a：Orion 的 HP SLO 达成率比 REEF 低 5.4×、比独占低 8.1×；REEF 比独占差 1.5×；REEF 的 LP 吞吐（TPS）比 Orion 低 2.1×。图中柱高与折线值仅图，正文只给比值。
  - Fig.5b（HP TPOT 箱线图，纵轴约 10–20 ms，仅图）：Orion 下生成一个 token 的时间波动比 REEF 大 4.2×、比独占大 10.9×。论文没有说明"波动"用的是方差、IQR 还是极差。
- **归因**（§3，原文）："Orion marks a kernel as memory-bounded only when its average memory bandwidth utilization exceeds 60%. As a result, the selected compute-intensive kernels can still saturate the GPU global memory bandwidth when collocating with the Llama-8B kernels during the decoding phase, which is hugely memory-intensive"；第二个原因是 Orion 基于 multi-streams，配对的 compute/memory kernel 不保证同时开始、同时结束，LP 残留的访存密集 block 会与 HP 后续的访存密集 block 共跑（stranded block，Fig.4a），并占住 SM 推迟 HP 其他 block。
- **注意**：60% 阈值是对 Orion 设计的描述，论文**没有直接测量**共置时的 HBM 带宽数值；"带宽干扰"是从 TPOT 波动推断的。§6.1 对 LithOS 也给出同样归因（HBM 带宽与 L2 竞争），同样没有带宽计数器数据。

## 问题 4：开销与硬件假设

- **切分粒度**：分段 = 一波 block（受带宽限制的 kernel 更少），不是按时间切；400 µs 是 block 时长的经验上界（Fig.3b 99.999%）。平均分段 77 µs（§1）；平均抢占 139 µs（§4.3.1），121–165 µs（Fig.13a）。block 本身超过 400 µs 的 kernel 无法保证，作者引用编译器方法作为正交方案（§2.1）。
- **host 侧开销**：同步 API 约 5 µs + launch 6–7 µs（§4.3）；device queue 限 1 个 LP kernel；kernel-tick 依赖 kernel 时长可预测。仅切分时 LP 吞吐 −37%，三项优化后相对 vanilla −1.3%（§6.3）。调度器为 host 线程 + 异步 tick 线程（Algorithm 1）；论文没有报告 CPU 占用。profile 每个任务数秒，PTX 改写每个 kernel 一次（§5）。
- **HP 开销**：相对独占 SLO 达成率下降 < 1%（摘要、§6.1）；P99 TPOT 10.9 vs 10.6 ms（Fig.12a）。
- **硬件假设**：只在独立 GPU（A100 SXM4、L40s、H100）上评估；显存管理依赖 NVLink 与修改的 UVM 驱动；关闭 DVFS（§6.1）。全文没有提到 Jetson、集成 GPU 或与 CPU 共享内存带宽的情形（全文检索 Jetson / integrated 无结果）。§5 声称不依赖特定硬件指令、支持任意架构代际，但只测了 Ampere、Ada、Hopper。

## 作者自述的局限

- 拿不到 cuBLAS、cuDNN 等闭源库的 PTX，需替换为 CUTLASS（§5）。
- 跨 block 同步和 persistent kernel 关闭切分（§5）。
- 单个 block 超过 400 µs 的情况仍可能出现，取决于程序员（§2.1）。
- consolidation 在 HP 请求意外到达时会增加抢占延迟（§4.3，实测 < 0.7%）。
- 论文没有单独的局限性章节。

## 与本场景的前提对照

| 前提 | 论文 | 本场景（见 [survey.md](../survey.md) §0.2） | 是否成立 |
|---|---|---|---|
| 硬件 | 独立 GPU A100/L40s/H100，HBM 独占，NVLink 多卡 | Thor SoC，20 SM，LPDDR5X 与 CPU 共享约 273 GB/s | 不成立 |
| 高优先级任务 | 在线 LLM serving（Llama-8B / Yi-34B），独占利用率约 33% | 小模型执行器，37.5 ms，周期 200 ms | 不成立（方向相反） |
| 低优先级任务 | 离线批推理或训练，无时延要求 | 27B 规划器，每 5.33 s 一次，有周期要求 | 部分成立：都是 LLM，但本场景 LP 有时延要求 |
| SLO 形式 | 独占 P99 TTFT/TPOT，按比例统计达成率 | 严格截止期 200 ms（或 533 ms） | 不成立 |
| 抢占机制 | 拦截 Driver API + PTX 改写，LP 在 queue 中最多 1 个分段 | 可用 MPS 客户端优先级、stream 优先级、green context；规划器 kernel 是否有 PTX 未知 | 待定（判断：取决于规划器 kernel 是否可取得 PTX、是否用 cuBLAS） |
| CUDA graph | 只描述"可以"拆子图，未评估 | 判断：FlashRT 类部署常用 CUDA graph | 未验证 |
| 干扰来源 | 空间共享下 HBM 带宽与 L2 竞争；时间独占后消除 | SM 划分后剩余 decode 带宽干扰；另有 CPU 内存流量干扰 | 部分成立：GPU 内部带宽干扰同向；CPU 侧流量论文不涉及 |
| 数值一致 | 未讨论 | 规划器 prefill 切小块会改变输出 | 判断：按 block 切分不改变 prefill 分块大小，与"切 chunk 改变输出"是不同层次；但见机制 2 的数值注意点 |
| 时间尺度 | HP 每 token 约 10 ms，抢占 ~139 µs | 执行器 37.5 ms，截止期 200 ms | 判断：µs 级抢占远低于本场景需要的 ms 级裕量 |

## 可以借用的 / 不能借用的

- **可以借用（判断）**：
  - "block 时长短而稳定，可把 LP 的设备占用切到一波"这一观察，可在 Thor 上对规划器 GEMM 直接测量（20 SM 时一波的时长与 block 时长相同量级，但 kernel 波数更多）。
  - 时间上完全不重叠（HP 运行时 LP 不在设备上）是消除 GPU 内部带宽干扰的直接手段，与本场景"SM 划分后剩余干扰来自 decode 带宽"一致；代价是规划器只能用执行器空闲时间。
  - device queue 只保留 1 个 LP kernel + 按预测时长定时下发（kernel-tick），是在没有硬件抢占时界定抢占延迟的通用做法，不依赖 MPS。
  - 用 host 侧 API 模式识别 HP 内部小气泡的方法，对执行器这种固定流程的周期任务更容易（执行器流程固定，可离线标定）。
- **不能直接借用（判断）**：
  - 论文的 LP 没有时延要求，调度策略只让 LP 吃气泡；本场景规划器有 5.33 s 周期，需要另外的保证。
  - CPU 内存流量对 Thor 共享带宽的干扰不在论文范围内，时间独占也消除不了。
  - NVLink 分层显存管理与本场景无关（统一内存 128 GB）。
  - 依赖 PTX 与 ptxas 重汇编；若规划器 kernel 来自闭源库或无 PTX 的 cubin，或依赖 CUDA graph，需要额外工作；论文对 graph 只有一句描述。

## 待核实

- Fig.10、Fig.11、Fig.13、Fig.14 的柱高只能从图读，本笔记未逐柱读数。
- Fig.12a 的 REEF TPOT：正文 12.5 ms，图中标注 13.5 与 12.4，无法对应。
- Fig.13a 每栏代表 HP 还是 LP 未说明。
- PTX 改写是否修正 `%nctaid`、如何处理依赖 gridDim 或 block 间 atomic 的 kernel，正文与补充材料均未说明。
- HP 与 LP 是否必须在同一进程或同一 CUDA context 内由同一调度器拦截，论文未说明。
- 代码是否开源：论文中未见代码链接。
- 检索：arXiv abs 页（2601.04071，v1/v2，无 Comments），PDF v2 全文（含补充材料 §A、§B）经 pdftotext 抽取阅读。
