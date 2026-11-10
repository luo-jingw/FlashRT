# 单 GPU 上 LLM 规划器与实时执行器共置：相关系统文献

范围：两个模型共享一块 GPU 时的抢占与优先级调度、SM 与显存带宽划分、LLM 阶段 / 多 LLM 共置、边缘多 DNN 实时调度与 DVFS，以及机器人快慢双系统是否测量了同卡干扰。

来源与核实方式：每条都打开过 arXiv abs / html、会议页（USENIX、PMLR、SIGOPS、Dagstuhl、CMU RI）或作者托管的 PDF，核对标题、ID 与会议；数字取自正文或摘要原文。只在检索片段中见到的细节标【仅摘要】。机构只写单位。

## 1. 抢占式 / 优先级 GPU 调度

| 简称 | 标题 | arXiv / 会议 | 机构 | 机制 | 关键结果 |
|---|---|---|---|---|---|
| REEF | Microsecond-scale Preemption for Concurrent GPU-accelerated DNN Inferences | OSDI'22 | 上海交通大学 IPADS；上海人工智能实验室 | 利用 DNN kernel 的幂等性，reset 式抢占：直接杀掉 BE kernel 再重跑；dynamic kernel padding 让 BE kernel 只占 RT kernel 剩下的 CU | AMD GPU 上 RT 端到端延迟开销 < 2%，吞吐比 RT 独占最高 7.7×；NVIDIA 上只有受限版本 |
| Paella | Paella: Low-latency Model Serving with Software-defined GPU Scheduling | SOSP'23 | University of Pennsylvania；DBOS | 编译器插桩 kernel，与客户端、调度器协同，由软件决定 kernel 下发顺序，避开硬件 FIFO 的队头阻塞 | 调度算法单项：负载 1.4×、延迟低 1.35×；对 Triton：负载最多 58×，延迟最多低 11× |
| Orion | Orion: Interference-aware, Fine-grained GPU Sharing for ML Applications | EuroSys'24 | ETH Zurich | 拦截 kernel launch 按算子调度；仅当 BE 与 HP kernel 资源画像互补（compute-bound 对 memory-bound）且 BE kernel 足够短时才共跑 | 与 BE 推理共置：吞吐最高 7.3×，HP p99 平均只比独占高 15% 以内 |
| Tally | Tally: Non-Intrusive Performance Isolation for Concurrent Deep Learning Workloads | arXiv 2410.07381；ASPLOS'25 | University of Toronto；Vector Institute；CentML | 把 BE kernel 变换为 slicing 或 persistent-thread-block 可抢占形式，线程块粒度调度，HP 优先 | 与训练共置时 HP P99 平均开销 7.2%（TGS 188.9%），吞吐为 TGS 的 80% 以上（A100） |
| Bless | Improving GPU Sharing Performance through Adaptive Bubbleless Spatial-Temporal Sharing | EuroSys'25 | 上海交通大学；Microsoft；Alibaba | 时空混合共享，按配额精确分配算力并细粒度调度 kernel，挤掉空闲 bubble；面向配额，不针对截止期 | 平均延迟比 SOTA 低 21.1–37.3% |
| GPreempt | GPREEMPT: GPU Preemptive Scheduling Made General and Efficient | USENIX ATC'25 | 清华大学；中国人民大学 | 基于时间片 yield 的上下文切换式抢占，不要求 kernel 幂等；hint 提前抢占，与数据准备重叠 | 抢占延迟 40 µs 以内，效果接近 LC 独占（USENIX 摘要页数字） |
| XSched | XSched: Preemptive Scheduling for Diverse XPUs | OSDI'25 | 上海交通大学 IPADS | 可抢占命令队列 XQueue 与三级硬件模型（撤回未下发命令 / 撤回已下发未执行命令 / 中断正在执行的命令，NVIDIA 上基于 TSG 与 trap handler），shim 透明拦截 | 固定优先级下 HP P99 最多降 2.10×；Jetson Orin：前台在 DLA、后台在 iGPU 跑耗带宽任务，只调度 NPU 时前台 P99 仍是独立运行的 1.67× / 1.55×，NPU 与 GPU 统一按优先级抢占后为 1.18× / 1.09× |
| LithOS | LithOS: An Operating System for Efficient Machine Learning on GPUs | arXiv 2504.15465；SOSP'25 | Carnegie Mellon University；Meta | TPC 粒度空间调度；透明 kernel atomization（切小 kernel，减少队头阻塞，kernel 中途可重分配 TPC）；硬件 right-sizing；透明 DVFS | 推理叠加：尾延迟比 MPS 低 13×，比此前最好系统低 3×，吞吐 1.6×；DVFS 平均省能 26%（最高 46%），P99 平均 +7%；未分析显存带宽争用 |
| Hummingbird | Hummingbird: SLO-Oriented GPU Preemption at Microsecond-scale | arXiv 2601.04071（预印本） | 清华 AIR；国科大；华中科技大学；蚂蚁集团；上海交通大学；北京大学 | 在 PTX 层按 SM 数、occupancy、带宽切分 LP kernel；HP 到达即停止下发 LP，设备队列限制在约 1 个切分 kernel | HP SLO 达成率为空间共享（Orion / LithOS）的 9.7×、时间共享（REEF）的 3.5×（摘要；正文平均值为 5.6× 与 3.0×）；平均抢占延迟 139 µs |

## 2. 空间划分与显存带宽划分 / 节流

| 简称 | 标题 | arXiv / 会议 | 机构 | 机制 | 关键结果 |
|---|---|---|---|---|---|
| FGPU | Fractional GPUs: Software-based Compute and Memory Bandwidth Reservation for GPUs | RTAS'19 | Carnegie Mellon University | 计算划分到 SM 子集；逆向内存层次后用 page coloring 划分 L2 与 DRAM | 最坏情况下共跑使读写事务慢 10× 以上；多租户下运行时间平均波动从 MPS 的 135% 降到 9% |
| libsmctrl | Hardware Compute Partitioning on NVIDIA GPUs | RTAS'23 | UNC Chapel Hill | 利用硬件调度器未公开的 TPC mask，按 stream 或 kernel 限定 TPC；用户态库，含 Jetson TX1 / TX2 / Xavier | YOLOv2：第 3 个 TPC（+50% 容量）只提速 14%，分区大小可作为优先级的代理 |
| SGDRC | SGDRC: Software-Defined Dynamic Resource Control for Concurrent DNN Inference on NVIDIA GPUs | arXiv 2407.13996；PPoPP'25 | HKUST；CUHK；Alibaba；MSR；上海交通大学 等 | 逆向 VRAM 通道哈希，在 nvidia-uvm 中用 shadow page table 做 cache coloring，把任务映射到通道子集；LS 队列空时 BE 用全部通道；算力侧 tidal SM masking | 仅通道隔离：LS kernel p99 平均降 28.7%（P40）/ 47.5%（RTX A2000）；SLO 达成率平均 99.0%，总吞吐最高 1.47×；只在 GDDR 独显上做过 |
| 边缘隔离实测 | Peformance Isolation for Inference Processes in Edge GPU Systems（原标题拼写如此） | arXiv 2601.07600（预印本） | Universitat Politècnica de València | A100 上比较 MPS 与 MIG；Jetson Orin Nano / AGX Orin 上比较 MPS 与 Green Contexts | MIG 隔离好；Green Contexts 无可测吞吐开销但没有内存隔离；Orin Nano 两模型并行时 GPU 频率降到约最大值一半（约 20 W）；AGX Orin 锁频 1.02 GHz 后超时大幅减少（未给百分比） |
| RT-Gang++ | Analysis and Mitigation of Shared Resource Contention on Heterogeneous Multicore: An Industrial Case Study | arXiv 2304.13110 | University of Kansas | Jetson Nano 上分区实时 gang 调度、LLC 带宽节流，用内存控制器 32 级可编程节流限制 iGPU 带宽 | 节流到约第 15 级前 DNN 不受影响；第 31 级时 SLAM 近乎完全隔离，但 DNN 延迟翻倍到约 73 ms；第 20 级是仍保住 20 Hz 的最强节流 |
| BWLOCK++ | Protecting Real-Time GPU Kernels on Integrated CPU-GPU SoC Platforms | arXiv 1712.08738；ECRTS'18 | University of Kansas | RT GPU kernel 执行期间节流高带宽 CPU 共跑任务 | Jetson TX2 上高带宽 CPU 共跑者最多使 GPU kernel 慢 3× |

## 3. LLM 阶段共置、多 LLM 共置、VLA 两阶段同卡

| 简称 | 标题 | arXiv / 会议 | 机构 | 机制 | 关键结果 |
|---|---|---|---|---|---|
| MuxServe | MuxServe: Flexible Spatial-Temporal Multiplexing for Multiple LLM Serving | arXiv 2404.02015；ICML'24 | CUHK；上海人工智能实验室 等 | 按热度共置多个 LLM，prefill / decode 拆成作业，用 MPS 给各进程分 SM，统一 KV 池 | 吞吐最高 1.8×；99% SLO 下可处理请求最多 2.9× |
| NanoFlow | NanoFlow: Towards Optimal Large Language Model Serving Throughput | arXiv 2408.12757；OSDI'25 | University of Washington；清华大学 等 | nano-batch，让 compute-bound GEMM 与 memory-bound attention / GEMV 同卡重叠；线程块数控制份额，成对干扰画像 + MILP 求调度 | 比 vLLM 等平均 1.91× 吞吐，达到最优吞吐的 50–72% |
| semi-PD | semi-PD: Towards Efficient LLM Serving via Phase-Wise Disaggregated Computation and Unified Storage | arXiv 2504.19867（预印本） | 清华大学；无问芯穹；上海交通大学 | prefill / decode 两个 worker 按 MPS 百分比分 SM，常驻进程持有权重与 KV，SLO 感知动态划分 | 平均端到端延迟低 1.27–2.58×（DeepSeek） |
| Bullet | Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration | arXiv 2504.19516；ASPLOS'26 | 中山大学 | prefill 与 decode 并发，libsmctrl stream SM mask 分区（重配约 4.1 µs）；SM-scaling roofline 模型按共跑时实测可达带宽校准 decode | 吞吐平均 1.26×；TTFT 比 SGLang-2048 低 4.2×；称 Green Contexts 额外占用 700 MB 以上内存 |
| MuxWise（v1 名 Drift【仅摘要】） | Towards High-Goodput LLM Serving with Prefill-decode Multiplexing | arXiv 2504.14489；ASPLOS'26 | 上海交通大学；NUS | Green Contexts 把 stream 绑到 SM 子集，prefill 按层切分，decode 优先；抗争用估计器在最坏带宽争用下为 decode SLO 定分区 | goodput 平均 2.20×（最高 3.06×）；按层 prefill 开销 ≤ 1.5% |
| DuetServe | DuetServe: Harmonizing Prefill and Decode for LLM Serving via Adaptive GPU Multiplexing | arXiv 2511.04791（预印本） | USC；UC Riverside | 默认聚合执行，roofline 预测 TBT 超限时才用 libsmctrl 开启 SM 空间复用 | 吞吐最高 1.3×；未量化分区间带宽干扰 |
| Robion | Efficient Vision-Language-Action Management and Serving for Robot Factories | arXiv 2609.12075（预印本） | MPI-SWS | 同卡把 VLA 的 VLM 阶段与 action DiT 阶段拆为两个 stream，VLM 用 Green Contexts 限 SM，按剩余延迟预算排优先级 | π0、20 台机器人 SLO 达成率：单体 0%、阶段拆分 56%、固定 SM 划分 67%、带保护的动态划分 82%；假设 VLM compute-bound，未测带宽干扰；在 RTX 6000 Pro / H100 上运行 |

## 4. 边缘多 DNN 实时调度与 DVFS

| 简称 | 标题 | arXiv / 会议 | 机构 | 机制 | 关键结果 |
|---|---|---|---|---|---|
| Miriam | Miriam: Exploiting Elastic Kernels for Real-time Multi-DNN Inference on Edge GPU | arXiv 2307.04339；SenSys'23 | CUHK；香港城市大学 | 源到源变换生成 grid / block 可调的 elastic kernel，用 BE kernel 分片填满关键 kernel 剩余资源（RTX 2060、Jetson AGX Xavier） | 吞吐 +92%，关键任务延迟开销 < 10%（摘要；正文部分负载 21–28%） |
| Fulcrum | Fulcrum: Optimizing Concurrent DNN Training and Inferencing on Edge Accelerators | arXiv 2509.20205（预印本） | Indian Institute of Science | 认为 Jetson 原生时分与空分保不住推理延迟，在应用层交错训练与推理 minibatch，并在约 18,000 个 power mode 上搜索 | Jetson AGX Orin 上 97% 以上的运行同时满足延迟与功耗预算 |
| Twill | Twill: Scheduling Compound AI Systems on Heterogeneous Mobile Edge Platforms | arXiv 2507.00491；ICCAD'25 | University of Turku；Politecnico di Milano | GPU / DLA / CPU 间亲和映射与迁移，按优先级冻结任务，DVFS 控制器在 10 W TDP 内调频 | 相对各基线平均延迟降 19–34%（最大 54%，Jetson Orin NX） |

## 5. 机器人快慢双系统是否测量同卡干扰

| 系统 | 来源 | 部署 | 测干扰 |
|---|---|---|---|
| Helix | Figure AI 技术博客（2025） | S2 7B VLM 7–9 Hz，S1 80M 200 Hz，各占一块嵌入式 GPU | 否，用两块硬件回避共享 |
| GR00T N1 | arXiv 2503.14734 | S2 在 L40 上 10 Hz，S1 DiT 120 Hz | 否，未说明是否同卡并发 |
| Fast-in-Slow | arXiv 2506.01953 | 快系统嵌入慢系统 VLM，异步 1:4；论文写明硬件不支持两系统在不同 GPU 上并行，在 4090 上串行 | 否 |
| RoboDual | arXiv 2410.08001 | OpenVLA-7B 加约 20M DiT 专才，异步，15 Hz | 否 |
| HiRT | arXiv 2410.05273；CoRL'24 | InstructBLIP-7B 低频异步加轻量策略，9.8 Hz | 否 |
| DualVLN on Jetson Thor | NeurIPS'26 On-Device Intelligence workshop poster | 7B 规划器 + diffusion 轨迹模型在单块 Thor 上，TensorRT + NVFP4 使规划中位延迟降 6.8× | 否 |

## 6. 对"带宽受限的 LLM decode + 截止期约束的扩散执行器"的适用性

- 算力侧隔离已有成熟手段（libsmctrl、Green Contexts、LithOS 的 TPC 调度）。真正划分 DRAM 带宽的只有 FGPU 与 SGDRC（逆向通道哈希 + page coloring），只在 GDDR 独显上验证过；Thor 的 LPDDR 统一内存通道映射与 QoS 仲裁没有公开文档。静态着色还会按比例削减规划器 decode 速度。
- 按时间窗限制带宽方有直接证据（XSched 在 Orin 上把耗带宽的 iGPU 后台纳入抢占后前台 P99 从 1.55–1.67× 回到 1.09–1.18×；RT-Gang++ 用内存控制器节流；BWLOCK++ 在 RT kernel 窗口内节流共跑者），但没有用在 LLM 规划器 + 机器人执行器上。自回归 decode 由一串短 kernel 组成，可以在 token / 推测周期边界让步。
- 干扰感知的分区定量（MuxWise、Bullet、Orion）可以改造成"在规划器 decode 的最坏带宽压力下为执行器定 SM 份额"；这些工作里 decode 是被保护方，这里 decode 是干扰源。Robion 是唯一在同卡上用 Green Contexts 划分 VLM 与 action DiT 的系统，但假设 VLM 为 compute-bound，对自回归 decode 规划器不成立。
- prefill 侧的抢占与 kernel 切分（REEF、GPreempt、Hummingbird、Tally、LithOS atomization、MuxWise 按层 prefill）能缩短阻塞，评测基本在 A100 / H100 上，只有 XSched 测过 Jetson Orin。
- DVFS / 功耗：边缘隔离实测发现 Orin Nano 两模型并行时 GPU 频率降到约一半；还没有工作把功耗上限导致的降频、带宽争用与执行器截止期放进同一个模型。
