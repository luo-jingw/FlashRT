# XSched：用"可抢占命令队列 + 三级硬件模型"在 host 侧统一调度各类 XPU

- **XSched** · 2025 · OSDI · [链接](https://www.usenix.org/conference/osdi25/presentation/shen-weihang)
- XSched: Preemptive Scheduling for Diverse XPUs
- 作者：Weihang Shen, Mingcong Han, Jialong Liu, Rong Chen, Haibo Chen（上海交通大学 IPADS）；Proceedings of the 19th USENIX OSDI，2025-07-07 至 07-09，pp. 671 起。元数据依据：usenix.org 下载的论文 PDF 首页会议印记与页脚（osdi25-shen-weihang.pdf，23 页含封面与参考文献）。
- 代码：公开，https://github.com/XpuOS/xsched（论文 §11 原文给出；2026-11-17 查询 GitHub API：Apache-2.0，未归档，最近推送 2026-11-13）。

## 论文做了什么

1. 提出 XQueue：一个可抢占的命令队列抽象，接口只有 submit / wait / suspend / resume（Table 1），类比 CPU 线程。
2. 提出三级硬件模型：Lv1 抢占还没下发的命令（pending），Lv2 抢占已下发、未执行的命令（in-flight），Lv3 抢占正在执行的命令（running）（§4，Fig.3，Table 2）。
3. 实现 XSched 框架：LD_PRELOAD 式的 shim 拦截驱动 API，progressive command launching 控制在途命令数，daemon 运行策略（§5）。
4. 在 7 个软件平台、10 种 XPU 上实现并评测固定优先级与带宽划分两个策略，外加 Jetson Orin / Intel Core Ultra 上跨 XPU 统一优先级，以及三个案例（多租户容器、视频会议、Triton/Paella）（§7、§8）。

## 结构

- §1 Introduction：硬件调度器 FCFS/RR 不够用；已有方案不可移植。
- §2 Background and Motivation：XPU 任务=命令序列；硬件调度与已有 host 调度方案的不足。
- §3 XQueue 抽象：接口（Table 1）与策略示例（Fig.2：固定优先级、带宽划分）。
- §4 Multi-level Hardware Model：Lv1/Lv2/Lv3 定义、stalling/flushing 两种 deactivate 方式、接口（Table 2）。
- §5 XSched 框架：XShim、XPreempt、XAL、XScheduler；progressive command launching（Fig.6）。
- §6 各 XPU 上的实现：Lv1 通用实现；Lv2 Intel NPU stalling、NVIDIA GPU 二进制插桩 flushing；Lv3 TSG-based 与 queue-based。
- §7 评估：可移植性（Table 3）、统一策略（Fig.9、Fig.10）、可演进性（Fig.11）、开销（Fig.12）。
- §8 案例：容器 GPU 收割（Fig.13）、AI PC 视频会议（Fig.14）、多模型推理服务（Fig.15）。
- §9 Discussion and Limitations；§10 Related Work；§11 Conclusion。

## 问题表述

- **场景**：
  - 硬件：NVIDIA GV100、K40m（独显）；AMD MI50；Intel Arc iGPU、Intel NPU3720（Core Ultra 9 185H AI PC）；Ascend 910b；NVIDIA DLA、PVA、OFA（Jetson Orin 上的加速器）；Xilinx VU9P FPGA（§7.1）。另在 Table 3 / §7.1 列出 Lv1 在 RTX/GTX 上的同步开销测试（Fig.5b：GTX 1060、RTX 2070、RTX 3080）。
  - 负载：同一 XPU 上两个独立进程，一前台一后台，提交同类任务；GPU/NPU 上都是 ResNet-152 推理；ASIC 上是 Stereo Disparity（OFA）、Gaussian Filter（PVA）；FPGA 上是向量加（§7.2）。命令时长 GV100 上 2–113 µs，DLA 上 95–661 µs（§7.2）。
  - 截止期：主体评估没有截止期，指标是前台 P99 相对单独运行的倍数；视频会议案例有周期约束（LFBW 25 FPS，whisper.cpp 每 3 s 一段，§8 Case 2）；Paella 对比用 K-EDF（§8 Case 3）。
- **优化目标**：论文没有给形式化目标函数；§1 表述为满足"priority and fairness"等多样调度需求，并要求通用性（Portability / Uniformity / Evolvability，§1）。
- **指标**：前台任务 P99 时延（相对单独运行，Fig.9 上、Fig.10）；带宽划分下前后台归一化吞吐（Fig.9 下）；P99 抢占延迟（Fig.11a/b）；运行时开销与单核 CPU 占用增量（Fig.11c、Fig.12）；案例中的帧时延、作业归一化性能、吞吐-时延曲线。

## 核心观察

1. 硬件原生调度对前后台一视同仁，单个后台任务就让前台 P99 变成单独运行的 1.60–2.19×。
   证据：§7.2 与 Fig.9（上），10 种 XPU，前台以 20 % 峰值吞吐周期发请求、后台满速发请求。
2. 同步下发（每个命令后 sync）作为抢占手段代价过高，且 XPU 越快越贵。
   证据：Fig.5(a)，ResNet-152 等任务在各 XPU 上开销 8.2 %–151.3 %；Fig.5(b) 从 GTX 1060 换到 RTX 3080，开销平均增加 3.04×（§5.2）。
3. 抢占延迟由实现层级决定：Lv1 ≈ 在途阈值 × 命令时长，Lv2 ≈ 一个命令时长，Lv3 与命令时长无关。
   证据：Fig.11(a)，被抢占任务持续下发 T=0.5 ms 的命令，在途阈值 8：Lv1 P99 ≈ 8T，Lv2 ≈ 1T，GV100 Lv3 为 32 µs（§7.3）；Fig.11(b) 在 GV100 上扫命令时长 0–2 ms，Lv1 线性增长、Lv2 斜率更小、Lv3 平坦（仅图）。
4. 在 SoC 上，即使 NPU 与 GPU 计算资源独立，GPU 上的访存后台任务仍通过共享内存带宽和供电拖慢 NPU 前台任务；只调度 NPU 不够，需要跨 XPU 统一优先级。
   证据：§7.2 "Heterogeneous priority policy" 与 Fig.10，见下文"Jetson Orin 实验"。
5. Lv1 的开销来自 progressive launching 引入的额外同步，阈值越大、命令越长，开销越小。
   证据：Fig.11(c)，GV100 Lv1，命令 0.01 / 0.1 / 1 ms；阈值 > 10 时开销 < 1 %（§7.4 正文）；实验中阈值取开销 < 3 % 的最小值（§7.4），各设备 2–16（Fig.9 图注）。

## challenge

- XPU 抢占能力差异大且在演进：GPU 可编程、主流 NPU/ASIC 不可编程；Tegra 的 timeslice ioctl 在桌面/服务器 GPU 上没有（§1）。
- 软件栈（用户态/内核态驱动、固件）复杂且定制化，调度机制与策略容易和具体硬件耦合（§1、§2.4）。
- 只用 Lv1 接口时，命令一旦 launch 就脱离 host 控制，既要低抢占延迟又不能破坏异步流水（§5.2）。

## 机制

### XQueue 与 XSched 框架（§3、§5）

- XShim：拦截驱动 API（如 cuLaunchKernel、cuMemcpyAsync），把命令重定向到 XQueue；可选，应用也可直接调 XQueue 接口（§5.1）。
- XPreempt：每个 XQueue 一个 worker 线程、pending 缓冲、in-flight 日志；agent 通过共享内存 IPC 与 XScheduler daemon 通信（§5.1、§5.3）。
- Progressive command launching（Fig.6）：在途命令数达到阈值时，sync 到在途命令的中间那个（等一半完成）后再继续 launch；suspend 时 worker 停止 launch；Lv2 时再 deactivate，Lv3 时再 interrupt。
- 策略：固定优先级 104 行、带宽划分 200 行 C++（§7.2）；用户通过 send_hint / XCLI 设置优先级、带宽、截止期等（§5.3）。

### NVIDIA GPU 上 Lv1/Lv2/Lv3 各需要什么（对应问题 1）

| 层级 | NVIDIA GPU 上的实现 | 依赖 | 论文中实现/测过的 GPU |
|---|---|---|---|
| Lv1 | 拦截 CUDA driver API，命令放进 host 缓冲，按阈值逐步 launch 到 CUstream；sync 用 command 后记录的 CUevent（§6.1） | 只需公开 driver API 与 API 拦截 | 所有 CUDA GPU；Table 3：Lv1 共享实现 318 行 |
| Lv2（flushing） | 动态二进制插桩：把每个 kernel 第一条指令改写为 JMP 到 per-kernel helper，helper 从 constant memory 读取 per-hwQueue 标志、hwq_id、cmd_id 并调用 guardian；标志置位时 kernel 记录自己的 ID 并立即退出；reactivate 时清标志并重新 launch 被中止的 kernel（§6.2，Fig.7、Fig.8） | SASS 级插桩（NVCC 编译 guardian 并在进程启动时装入 instruction memory）；用 CUDA export table 中的隐藏函数分配和访问 GPU instruction / constant memory（§6.2）；被 flush 的 kernel 要能从头重跑 | Kepler、Volta、Ampere（§7.1、Table 3；DBI 共享 513 行） |
| Lv3（TSG-based） | interrupt 把要被抢占的 hwQueue 所在 TSG 的 timeslice 设为 0，restore 恢复原值；GPU 在 timeslice 到期时中断该 TSG 所有正在运行的 kernel 并轮转到下一个 TSG（§6.3） | 调整 TSG timeslice 的驱动能力（引 [80] open-gpu-doc 的 GV100 dev_ram 文档与 [123]）；一个进程=一个 CUDA context=一个 TSG，因此只能做进程间调度；该方法同时达到 Lv2 与 Lv3 | "NVIDIA GPUs released after the Pascal architecture"（§7.1）；Table 3 "NV GPUs w/ TSG" 90 行；Fig.9 中只在 GV100 上作为 XSched-T 出现 |
| Lv3（queue-based） | 通过跟踪 CUDA 应用的 syscall 发现一个 ioctl，写 GPU 控制寄存器触发中断（原用途是 kernel 调试）；中断后所有正在运行的 kernel 停下并进入 trap handler；用 DBI 扩展 trap handler 识别 XSched 发起的中断，目标 kernel 被中止、其余恢复上下文继续（§6.3） | 未公开的 ioctl + SASS 级插桩扩展 trap handler；GPU 不支持从中断点恢复被中止的 kernel，所以只抢占幂等 kernel 并从头重跑；幂等性目前人工标注（§6.3 脚注 4） | Volta、Ampere（§7.1）；Fig.11(a) 只给了 GV100 的 Lv3 数字 |

- Jetson/Tegra 上展示了什么：论文自己的 Jetson Orin 实验只涉及 DLA（CUDLA 平台，Lv1，Table 3 与 Fig.9 "DLA/Lv1"）、PVA 与 OFA（VPI 平台，Lv1），以及 Orin 集成 GPU 上的低优先级访存后台任务（Fig.10）。论文没有说明 Orin GPU 上用的是哪一级；没有任何 Lv2 或 Lv3 在 Tegra GPU 上的结果。
- TSG timeslice 与 Tegra：§1 与 §2.4 指出 [123]（Wang, Liu, Wong, Kim, "Unleashing the Power of Preemptive Priority-based Scheduling for Real-Time GPU Tasks", arXiv 2401.16529）在 NVIDIA Tegra 嵌入式 GPU 上通过特定 ioctl 调整 TSG timeslice 实现抢占，§1 称该 ioctl "is not available in desktop or server GPUs"；XSched 在 GV100 等桌面/服务器 GPU 上实现了"similar approach"（§6.3）。论文没有说明它在 GV100 上用什么接口改 timeslice，也没有在 Tegra 上运行 TSG-based 方案。
- 对 Thor 是否适用：论文未涉及 Thor 或 Blackwell。判断：TSG timeslice 是 NVIDIA GPU host 调度（runlist）的通用概念，Thor 的集成 GPU 在硬件上大概率仍有 TSG；但能否从用户态调整取决于 Thor 所用驱动是否暴露 [123] 那种 Tegra ioctl 或 XSched 在 GV100 上用的接口，需要在 Thor 上实测。另一点判断：TSG 粒度是 CUDA context；本场景若两个模型都作为 MPS 客户端运行，它们共享 MPS server 的 context，TSG-based 抢占无法区分执行器和规划器；不用 MPS 时两进程各自一个 TSG，才符合该方案的前提。

### Lv2 stalling（Intel NPU，§6.2）

- 修改 Intel NPU 驱动，暴露 2024-07 新固件中微控制器（LeonRT）暂停取命令的能力；无额外运行时开销（§7.4）。

## 评估

- **基线**：各 XPU 原生硬件调度（Native）、单独运行（Standalone）；案例中 vCUDA、TGS（容器）、Triton 原生与 Triton 优先级设置、Paella（推理服务）；还有去掉 progressive launching 的 XSched w/o prog（Fig.13）。
- **负载**：见"问题表述"；案例负载：PyTorch 训练（TGS 同款）、BlackScholes（生产）+ CFD（机会作业）；LFBW + whisper.cpp；两个 Bert-large（高优先级 10 req/s，低优先级满速）；Paella 的 log-normal（σ=2.0）负载。
- **主要结果**：
  - 固定优先级：前台 P99 为单独运行的 1.02–1.30×，相对原生最多降 2.11×（§7.2 正文；摘要与 §1 写 2.10×）。支持 Lv2/Lv3 的 GV100、K40m、NPU3720 上前台 P99 仅比单独运行差 1.9 %、3.8 %、5.4 %，仅 Lv1 的设备差 7.3 %–29.6 %（§7.3）。
  - 带宽划分（前台 75 %、后台 25 %）：总吞吐平均开销 1.5 %，比例达到设定（§7.2，Fig.9 下）。
  - 跨 XPU 统一优先级：前台 NPU P99 从只调度 NPU 时的 1.67× / 1.55× 降到 1.18× / 1.09×，相对原生最多降 2.63×（§7.2，Fig.10）。
  - 运行时开销：Lv1 在所有 XPU 上 < 3.4 %；Lv2 在 GV100 / K40m 上比 Lv1 再多 2.1 % / 4.0 %（guardian 读标志的延迟，HBM2 比 GDDR5 好）；NPU3720 硬件 Lv2 无额外开销（§7.4，Fig.12a）。单核 CPU 占用增加多数 < 5 %，910b 18.3 %、PVA 11.9 %（驱动 sync 自旋，§7.4，Fig.12b）。
  - 案例 1（GV100）：TGS 只能为机会作业收割 7.3 % GPU，XSched 20.0 %（2.74×）；生产作业退化 XSched 1.0 %，vCUDA 15.1 %（训练）/ 80.0 %（金融），TGS 在金融负载退化 70.0 %；MI50 上仅 Lv1 时生产作业退化 4.1 % / 0.4 %（§8 Case 1，Fig.13）。
  - 案例 2（NPU3720）：原生 FCFS 下 LFBW P99 帧时延 880 ms（单独运行的 20.12×，原因是等 whisper.cpp 的 0.8 s 任务）；固定优先级使 whisper.cpp 超出 3 s 周期；laxity-based 策略使 P99 降到 95 ms（9.26×）且 whisper.cpp 不丢内容（§8 Case 2，Fig.14）。
  - 案例 3（GV100）：Triton 集成 10 行代码，高优先级 Bert-large P99 从 1.53×（T+Priority 1.51×）降到 1.07×，比原生 Triton 低 30.0 %；Paella 对比集成 15 行，K-EDF（K=16），低吞吐时 P99 与 Paella 相当，1,000 req/s 时好 1.3×（§8 Case 3，Fig.15）。
- **消融**：Lv1/Lv2/Lv3 分层对比（Fig.11a、§7.3）；阈值与命令时长对开销（Fig.11c）；去掉 progressive launching（XSched w/o prog，Fig.13 右，仅图）。

### Jetson Orin SoC 实验（对应问题 2）

- 设置（§7.2 "The uniform policy for heterogeneous XPUs"）：先在 NPU 上跑与 Fig.9 上相同的负载——两个进程都做 ResNet-152 推理，前台周期发、后台满速——Orin 上的 NPU 是 DLA，Core Ultra 上是 NPU3720；再在另一个低优先级进程里用 SoC 的集成 GPU 跑"memory bandwidth-consuming tasks"。论文没有给出 GPU 任务的具体内容、Orin 的具体型号或功耗模式、以及任何 GB/s 带宽数字。
- 干扰数字只以前台 P99 相对单独运行的倍数给出：只用 XSched 调度 NPU（XSched NPU）时 P99 为 1.67× 和 1.55×；统一调度 NPU+GPU（XSched N+G）时 1.18× 和 1.09×；相对原生最多降 2.63×（§7.2）。正文没有逐一写明哪个数对应哪个平台；按正文列举顺序（NVIDIA DLA、Intel NPU3720）与 Fig.10 读图（Orin 单独运行 P99 约 3.6 ms、XSched NPU 约 6 ms、N+G 约 4.2 ms、Native 约 10–11 ms；Core Ultra 单独约 24 ms、XSched NPU 约 37 ms、Native 约 70 ms，仅图、目测），1.67× / 1.18× 对应 Orin，1.55× / 1.09× 对应 Core Ultra。
- 机制：统一优先级策略让高优先级 XQueue 抢占所有低优先级 XQueue，不管它们在哪个 XPU 上（§7.2）。前台 DLA 任务就绪时，GPU 上的后台 XQueue 也被 suspend，停止 launch 新命令，于是在前台运行期间不再产生 GPU 访存流量；这是时间上的互斥，不是带宽配额。剩余的 1.18× 可对应 GPU 上已在途命令的收尾（判断，论文未分解）。

## 作者自述的局限

- §9：(1) 只适用于由 host 下发命令的 XPU，DPU 或能自主执行的 FPGA 不适用；(2) **单命令任务**：Lv1 与 Lv2 只对多命令任务有效，"For tasks containing only a single command, such as CUDA graph or model inference on certain NPUs"，需要实现 Lv3 或通过 model slicing 把任务切成多个细粒度命令；(3) 假设显存足够，不管内存调度；(4) 恶意租户可绕过 XShim。
- §6.3：queue-based Lv3 只能抢占幂等 kernel 且从头重跑，幂等性人工标注。
- §7.3：GV100 上的 Lv3 依赖"undocumented and potentially unstable driver interface"，失效时需关闭 Lv3。

## 与本场景的前提对照

| 前提 | 论文 | 本场景（见 [survey.md](../survey.md) §0.2） | 是否成立 |
|---|---|---|---|
| 硬件 | 独显 GV100/K40m/MI50 等；SoC 为 Jetson Orin（DLA+GPU）与 Core Ultra（NPU+iGPU） | Jetson AGX Thor，Blackwell 集成 GPU 20 SM，统一内存约 273 GB/s | 部分成立：有统一内存 SoC，但 Orin 实验的前台在 DLA 上，GPU 只跑后台；没有 Blackwell / Thor |
| 两个任务所在引擎 | Orin 实验中前台 DLA、后台 GPU（不同引擎）；其余实验同一 XPU 上两进程 | 执行器和规划器都在同一 GPU 上 | 不成立（Orin 实验）；成立（Fig.9 同 XPU 实验，但不在 Tegra GPU 上） |
| 干扰来源 | Orin：计算独立，仅共享带宽与供电 | SM 不重叠后剩余干扰来自 decode 的带宽占用；不分 SM 时主要是 prefill 抢 SM | 部分成立：带宽干扰同构；但论文的缓解手段是暂停后台，不是带宽分区 |
| 后台任务形态 | ResNet-152 推理或未说明的合成访存任务，命令 µs 到亚 ms 级，多命令 | 27B LLM，CUDA graph 下发，一次 forward 约 1.17 s，decode 27 ms/token | 不成立：论文 §9 明确 CUDA graph 是单命令任务，Lv1/Lv2 失效 |
| 截止期 | 主体无截止期，指标是 P99 倍数；案例 2 有周期约束 | 执行器 200 ms 严格截止期，37.5 ms 单次 | 部分成立：目标同为前台尾时延，但无硬截止期分析 |
| 可抢占性 | Lv2 需 SASS 插桩（Kepler/Volta/Ampere）；Lv3 需 TSG timeslice 或未公开 ioctl（Pascal 之后，queue-based 仅 Volta/Ampere）；Lv3 queue-based 需幂等 kernel | Blackwell（Thor）；NVFP4 kernel，KV cache 原地写入 | 未知：论文未覆盖 Blackwell；仓库 README 把 Ampere 的 Lv2/Lv3 标为进行中、"Other NVIDIA GPUs"标为未实现 |
| 低优先级吞吐要求 | 后台为 best-effort，按优先级可被饿死；案例 2 用 laxity 策略保后台周期 | 规划器每 5.33 s 一次，L 有上界要求 | 部分成立：需要像案例 2 那样的 laxity 类策略，而非纯固定优先级 |
| 多进程隔离方式 | 不用 MPS；TSG-based 需每进程独立 context | 现有最佳方案是 MPS 客户端优先级 | 判断：MPS 下两个客户端共享 server context，TSG-based 不可区分；与 MPS 二选一 |

## 可以借用的 / 不能借用的

全部为判断。

- **在途阈值的权衡（对应问题 3）**：
  - 论文给出的关系：Lv1 抢占延迟 ≈ 在途阈值 N × 命令时长 T（Fig.11a，N=8 时 ≈ 8T）；Fig.6 中在途数达到 N 后等一半完成，所以在途数在 N/2 与 N 之间摆动。开销随 N 增大、T 增大而下降（Fig.11c，GV100：T=0.01 ms 时 N=1 开销 > 30 %、N=10 时约 2 %；T=0.1 ms 时 N=1 约 13 %、N≥2 约 2 %；T=1 ms 时 N=1 约 5 %、之后接近 0；均为仅图、目测）。论文没有给出解析公式，只有这两张图和"阈值 > 10 时开销 < 1 %"。
  - 规划器若整块 forward 用一个 CUDA graph 下发：按 §9，这是单命令任务，Lv1 的抢占延迟就是整个 graph 的时长，约 1.17 s（prefill 块）或 27 ms（decode 一个 token），远超执行器 37.5 ms 的单次时长；Lv2 flushing 依赖逐 kernel 插桩，论文未说明能否作用于 graph 内的 kernel。
  - 若把规划器 forward 切成每层或每几层一个 graph（相当于 §9 的 model slicing）：抢占延迟 ≈ N × 每段时长。每段时长 S 越短，抢占越快，但每段之间多一次 graph launch，且 progressive launching 每 N/2 段 sync 一次；规划器段时长是 ms 级，按 Fig.11c 的趋势，N=2 就足以把开销压到 1 % 左右，此时抢占延迟约 2S。要让这个延迟相对 200 ms 截止期可忽略，S 需要在几 ms 量级。注意本场景观察到的"切更小会改变输出"指的是 token 维度的 chunk 大小，按层切 graph 不改变数值（需核对实现）。
  - 若规划器不用 graph、逐 kernel 下发：Lv1 可用，抢占延迟 ≈ N × 单 kernel 时长（prefill GEMM ms 级、decode kernel µs 级），但失去 graph 带来的 launch 开销优化；decode 27 ms/token 里 launch 开销占比在 Thor 上未测。
  - 即便抢占及时，抢占只消除"规划器占着 GPU"这一部分；执行器运行期间如果规划器被完全暂停，规划器的 L 会被拉长，执行器每 200 ms 约 37.5 ms，规划器最多损失约 19 % 时间（未计切换开销）。
- **可借用**：
  - 跨引擎统一优先级的思路：带宽干扰可以通过"前台就绪时暂停后台下发"缓解，不需要硬件带宽分区；Orin 实验是该思路在 Jetson 上的唯一直接证据，但前台在 DLA。
  - Lv1 progressive launching 只需拦截公开 CUDA driver API，可移植到 Thor（前提：规划器按段下发）。
  - laxity-based 策略（案例 2）同时照顾前台帧率与后台周期，对应执行器截止期 + 规划器 5.33 s 周期的双约束。
  - 开源实现：README 称 2026-06 起支持 CUDA Graph 并"compatible with TSG-based Level-3 preemption"，2025-07 已集成 llama.cpp；这些是 README 自述，论文未评测。
- **不能直接借用**：
  - Lv2 flushing 与 Lv3 queue-based 依赖 SASS 级 DBI、CUDA export table 隐藏函数、未公开 ioctl，论文只在 Kepler/Volta/Ampere 上实现；Blackwell（sm_110 一类）没有证据。
  - Lv3 queue-based 要求被抢占 kernel 幂等；LLM 的 KV cache 写入、原地更新、NVFP4 量化中间写回等是否幂等需逐个 kernel 判定。
  - TSG-based Lv3 与 MPS 不兼容（同一 context）；放弃 MPS 就失去当前最好的 MPS 客户端优先级方案，需要实测比较。

## 待核实

- 1.67× / 1.18× 与 1.55× / 1.09× 分别对应 Orin 与 Core Ultra，是按正文列举顺序和 Fig.10 读图推定，正文未逐一指明；2.63× 出自哪个平台未写明。
- Orin 实验中 GPU 后台访存任务的内容、Orin 具体型号与功耗模式、GPU 侧实现的层级，论文均未给出。
- XSched 在 GV100 上调整 TSG timeslice 所用的具体接口（论文只引 [80] open-gpu-doc 与 [123]），以及 Thor 驱动是否提供同类接口。
- 仓库中 CUDA Graph 支持与 TSG Lv3 的具体实现方式、是否支持 Blackwell / Tegra，未读代码。
- 固定优先级的最大降幅：摘要与 §1 写 2.10×，§7.2 正文写 2.11×，论文内部不一致。
