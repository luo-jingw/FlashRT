# 经典系统论文如何陈述与论证核心贡献

范围：通用系统（OS、分布式系统、网络、存储）中 10 篇经典或获奖论文，以及 3 篇关于“如何写/如何评判系统论文”的短文。关注点是**论证结构**（问题如何被引出、洞察如何表述、贡献如何列出、评估如何对应主张），不是技术内容本身。

记号：
- `†` = 未核实（官方页面无法访问，或仅有二手来源）。
- “判断” = 本综述的归纳或评价，不是论文原文。
- 引文均为从论文 PDF 文本中逐字摘出；位置写为“Abstract / §n / 页码”。页码是会议论文集页码（PDF 页脚可见的才写，否则标 †）。引文中的参考文献编号（如 “[26]”）已删去，省略处用 “...” 标出；PDF 抽取造成的断词连字符已复原。

---

## 1. 选择标准与检索记录

### 1.1 选择标准

按以下顺序取舍：
1. **获奖可在官方页面核实**：USENIX 论文页面上的 “Awarded Best Paper” 标签；ACM SIGOPS Hall of Fame Award 名单（该奖专门授予发表 10 年以上、影响持久的 OS 论文）。
2. **领域内公认的奠基性论文**：引用量极高、被后续工作普遍当作基线或术语来源（引用量本身未逐一核实†）。
3. **论证类型多样**：刻意覆盖不同的贡献类型——新抽象（MapReduce）、基于工作负载观察的系统设计（GFS）、组合已知技术的生产系统（Dynamo）、新 API（Spanner）、新算法且以“可理解性”为目标（Raft）、打破权衡的 OS 架构（Arrakis、IX）、新机制且重新定义效率指标（Shenango）、刻画/bug 研究（Wasted Cores）、十年经验报告（Borg）。
4. 同一问题出现两篇获奖论文时（Arrakis 与 IX，同届 OSDI'14 最佳论文、同一问题“kernel 在数据路径上”），两篇都收入，用于对比“同一问题的两种框定方式”。

### 1.2 候选与取舍

| 候选 | 结论 | 理由 |
|---|---|---|
| MapReduce (OSDI'04) | 收入 | SIGOPS HoF 2015；新抽象型贡献的范本 |
| GFS (SOSP'03) | 收入 | SIGOPS HoF 2015；“观察→重审假设”型论证的范本 |
| Dynamo (SOSP'07) | 收入 | SIGOPS HoF 2017；明确把“组合已知技术”当作贡献 |
| Spanner (OSDI'12) | 收入 | OSDI'12 Jay Lepreau Best Paper；SIGOPS HoF 2022 |
| Raft (ATC'14) | 收入 | ATC'14 Best Paper；把“可理解性”设为首要目标 |
| Arrakis (OSDI'14) | 收入 | OSDI'14 Best Paper |
| IX (OSDI'14) | 收入 | OSDI'14 Best Paper |
| Shenango (NSDI'19) | 收入 | 未见获奖；以“latency vs CPU efficiency”框定问题的代表作，Caladan (OSDI'20) 的前作 |
| Linux Scheduler: Decade of Wasted Cores (EuroSys'16) | 收入 | 未查到获奖†；刻画/bug 研究型贡献的范本 |
| Borg (EuroSys'15) | 收入 | 未查到获奖†；经验报告型贡献及自定义评估指标（cell compaction）的范本 |
| Caladan (OSDI'20) | 未收入 | USENIX 页面无获奖标注；论证结构与 Shenango 高度重合 |
| FaRM (NSDI'14) | 未收入 | 已下载；与 Arrakis/IX 同属“新硬件趋势→新设计”，类型重复 |
| Snap (SOSP'19)、Exokernel、RAMCloud | 未收入 | 篇幅所限；类型与已收入论文重合（判断） |

### 1.3 检索与核实记录

- 论文全文来源：usenix.org 论文 PDF（Spanner、Raft、Arrakis、IX、Shenango、Caladan、FaRM）；research.google / static.googleusercontent.com（MapReduce、GFS、Borg）；allthingsdistributed.com（Dynamo，Werner Vogels 站点）；作者主页 people.ece.ubc.ca/sasha（Wasted Cores）。
- 获奖核实：
  - `https://www.usenix.org/conference/osdi14/technical-sessions` —— Arrakis、IX、Haven 三篇标注 “Awarded Best Paper”。
  - `https://www.usenix.org/conference/osdi12/technical-sessions/presentation/corbett` —— “Awarded Jay Lepreau Best Paper!”。
  - `https://www.usenix.org/conference/atc14/technical-sessions/presentation/ongaro` —— “Awarded Best Paper!”。
  - `https://www.usenix.org/conference/nsdi19/presentation/ousterhout`、`https://www.usenix.org/conference/osdi20/presentation/fried` —— 无获奖标注。
  - `https://www.sigops.org/awards/hof/` —— MapReduce、GFS（2015），Dynamo（2017），Spanner（2022），Lampson “Hints for Computer System Design”（2005）。Raft、Borg、Arrakis、IX 不在名单中。
  - EuroSys'15（Borg）、EuroSys'16（Wasted Cores）的最佳论文名单：检索未找到官方来源，**获奖情况未核实†**。
- ACM DL 页面对脚本返回 403，GFS 与 Dynamo 的 DOI 未能在官方页面核实†；Borg 与 Wasted Cores 的 DOI 取自论文 PDF 首页版权栏。
- 写作短文：Levin & Redell 取自 USENIX 转载页（页面注明原载 ACM SIGOPS OSR 17(3), July 1983, pp. 35–40）；Roscoe 取自 ETH 个人页面 PDF；Lampson 取自 arXiv 2011.02455（版本日期 May 14, 2021）。Lampson 1983 原版下载失败，未逐字阅读，仅凭 SIGOPS HoF 名单核实其存在与获奖。

---

## 2. 论文卡片

### 2.1 MapReduce

- **元数据**：Jeffrey Dean, Sanjay Ghemawat. “MapReduce: Simplified Data Processing on Large Clusters”. OSDI 2004. 链接：https://www.usenix.org/legacy/events/osdi04/tech/dean.html ；https://research.google/pubs/mapreduce-simplified-data-processing-on-large-clusters/ 。奖项：ACM SIGOPS Hall of Fame Award 2015（来源：sigops.org/awards/hof）。
- **一句话论点**：“The major contributions of this work are a simple and powerful interface that enables automatic parallelization and distribution of large-scale computations, combined with an implementation of this interface that achieves high performance on large clusters of commodity PCs.”（§1 末段）
- **引出问题的观察**：作者五年内写了数百个专用计算，“Most such computations are conceptually straightforward. However, the input data is usually large ... The issues of how to parallelize the computation, distribute the data, and handle failures conspire to obscure the original simple computation with large amounts of complex code to deal with these issues.”（§1）观察来自作者自身的重复劳动，而不是外部 benchmark。
- **关键洞察**：“We realized that most of our computations involved applying a map operation to each logical ‘record’ ... and then applying a reduce operation to all the values that shared the same key”；“Our use of a functional model with user-specified map and reduce operations allows us to parallelize large computations easily and to use re-execution as the primary mechanism for fault tolerance.”（§1）——形式即“计算是函数式的（无副作用），所以失败时可以直接重跑”。
- **贡献类型**：新抽象（编程模型/接口）+ 其实现。贡献列表是一句话两部分：interface + implementation（见上）。
- **评估如何支撑论点**：
  - “接口足够表达真实任务” → §2.3 举例 + §6 使用统计；摘要：“hundreds of MapReduce programs have been implemented and upwards of one thousand MapReduce jobs are executed on Google’s clusters every day.”
  - “实现性能好” → §5.3 1TB sort：“the entire computation takes 891 seconds. This is similar to the current best reported result of 1057 seconds for the TeraSort benchmark”。
  - “re-execution 足以容错” → §5.5 故障注入：“intentionally killed 200 out of 1746 worker processes”，仅 “an increase of 5% over the normal execution time”。
  - “backup task 机制必要” → §5.4 关闭该机制：“The entire computation takes 1283 seconds, an increase of 44% in elapsed time.”
  - “抽象降低编程复杂度” → §6.1 生产索引系统重写：“dropped from approximately 3800 lines of C++ code to approximately 700 lines”。
  - 判断：每个机制（容错、straggler 处理）都有一个“开/关”或“注入故障”的实验对应；“简单”这一主张用代码行数和使用量支撑，而不是用户研究。

### 2.2 GFS

- **元数据**：Sanjay Ghemawat, Howard Gobioff, Shun-Tak Leung. “The Google File System”. SOSP 2003 (Bolton Landing, NY). 链接：https://research.google/pubs/the-google-file-system/ ；DOI 10.1145/945445.945450†。奖项：SIGOPS Hall of Fame Award 2015。
- **一句话论点**：“our design has been driven by key observations of our application workloads and technological environment, both current and anticipated, that reflect a marked departure from some earlier file system design assumptions. We have reexamined traditional choices and explored radically different points in the design space.”（§1 第一段）
- **引出问题的观察**：四条编号观察（§1）：“First, component failures are the norm rather than the exception.” “Second, files are huge by traditional standards. Multi-GB files are common.” “Third, most files are mutated by appending new data rather than overwriting existing data. Random writes within a file are practically non-existent.” “Fourth, co-designing the applications and the file system API benefits the overall system by increasing our flexibility.” 规模数字：“The largest ones have over 1000 storage nodes, over 300 TB of disk storage”。
- **关键洞察**：每条观察直接推出一个设计取舍。例：“Given this access pattern on huge files, appending becomes the focus of performance optimization and atomicity guarantees, while caching data blocks in the client loses its appeal.”（§1 第三条）；“we have relaxed GFS’s consistency model to vastly simplify the file system without imposing an onerous burden on the applications.”（§1 第四条）——形式为“工作负载是 append-mostly，所以可以放松一致性、取消客户端缓存”。
- **贡献类型**：基于工作负载刻画的系统设计；没有编号贡献列表，摘要写为 “we present file system interface extensions designed to support distributed applications, discuss many aspects of our design, and report measurements from both micro-benchmarks and real world use.”
- **评估如何支撑论点**：
  - §6 开头明确评估目的：“present a few micro-benchmarks to illustrate the bottlenecks inherent in the GFS architecture and implementation, and also some numbers from real clusters in use at Google.” ——即评估目标包括**暴露架构自身的瓶颈**。
  - “故障是常态，所以自动恢复必须快” → §6.2 杀死一台 chunkserver：“All chunks were restored in 23.2 minutes, at an effective replication rate of 440 MB/s.”
  - “append 主导” → §6.3 Workload Breakdown（Appends versus Writes）用生产数据回头验证 §1 的观察。
  - 判断：GFS 的评估形成一个闭环——§1 的观察作为前提，§6.3 用生产 trace 再次确认前提成立。

### 2.3 Dynamo

- **元数据**：G. DeCandia, D. Hastorun, M. Jampani, G. Kakulapati, A. Lakshman, A. Pilchin, S. Sivasubramanian, P. Vosshall, W. Vogels. “Dynamo: Amazon’s Highly Available Key-value Store”. SOSP 2007 (Stevenson, WA), pp. 205–220†. 链接：https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf ；DOI 10.1145/1294261.1294281†。奖项：SIGOPS Hall of Fame Award 2017。
- **一句话论点**：“To achieve this level of availability, Dynamo sacrifices consistency under certain failure scenarios. It makes extensive use of object versioning and application-assisted conflict resolution in a manner that provides a novel interface for developers to use.”（Abstract）
- **引出问题的观察**：业务约束而非技术趋势：“customers should be able to view and add items to their shopping cart even if disks are failing, network routes are flapping, or data centers are being destroyed by tornados.”（§1）；“For many services ... the common pattern of using a relational database would lead to inefficiencies and limit scale and availability.”（§1）；SLA 以尾延迟衡量：“SLAs are expressed and measured at the 99.9th percentile of the distribution.”（§2.2）
- **关键洞察**：许多服务只需主键访问且可容忍短暂不一致，因此可以“always writeable”，把冲突留给读时、由应用合并。
- **贡献类型**：系统设计，且**显式把“组合已知技术并在生产中验证”定义为贡献**：“The main contribution of this work for the research community is the evaluation of how different techniques can be combined to provide a single highly-available system. It demonstrates that an eventually-consistent storage system can be used in production with demanding applications. It also provides insight into the tuning of these techniques to meet the requirements of production systems with very strict performance demands.”（§1）；方法来源也直言：“Dynamo uses a synthesis of well known techniques”（§1）。
- **评估如何支撑论点**：
  - “最终一致在生产可用” → §6.3 测量应用实际看到的分歧版本数：“99.94% of requests saw exactly one version; 0.00057% of requests saw 2 versions”。该节先承认完整刻画超出范围，然后选一个“good summary metric”。
  - “高可用” → §7：“applications have received successful responses (without timing out) for 99.9995% of its requests and no data loss event has occurred to date.”
  - “可调” → §6 的 (N,R,W) 配置与 99.9th 延迟曲线。
  - 判断：Dynamo 的评估不是与基线对比，而是**量化“牺牲的东西实际有多小”**——为“放弃一致性”这一有争议的取舍提供证据。

### 2.4 Spanner

- **元数据**：J. C. Corbett, J. Dean, M. Epstein, ... (26 位作者). “Spanner: Google’s Globally-Distributed Database”. OSDI 2012, pp. 251–264. 链接：https://www.usenix.org/conference/osdi12/technical-sessions/presentation/corbett 。奖项：OSDI'12 Jay Lepreau Best Paper（USENIX 论文页）；SIGOPS Hall of Fame Award 2022。
- **一句话论点**：“It is the first system to distribute data at global scale and support externally-consistent distributed transactions.”（Abstract）
- **引出问题的观察**：用户抱怨与用户行为：“we have also consistently received complaints from users that Bigtable can be difficult to use for some kinds of applications: those that have complex, evolving schemas, or those that want strong consistency in the presence of wide-area replication.” “Many applications at Google have chosen to use Megastore ... despite its relatively poor write throughput.”（§1，p. 251）——即“用户宁可接受低吞吐也要强语义”这一显示性偏好。
- **关键洞察**：“The key enabler of these properties is a new TrueTime API and its implementation. The API directly exposes clock uncertainty, and the guarantees on Spanner’s timestamps depend on the bounds that the implementation provides. If the uncertainty is large, Spanner slows down to wait out that uncertainty.”（§1，p. 252）——形式为“时钟不确定性有界且可测，所以等待它即可得到外部一致性”。
- **贡献类型**：系统设计 + 新抽象（时间 API）。无编号列表；结论把贡献收束为一个可迁移的设计原则：“the linchpin of Spanner’s feature set is TrueTime. We have shown that reifying clock uncertainty in the time API makes it possible to build distributed systems with much stronger time semantics.” 并给出对社区的主张：“As a community, we should no longer depend on loosely synchronized clocks and weak time APIs in designing distributed algorithms.”（§8 Conclusions）
- **评估如何支撑论点**：
  - “代价可接受” → §5.1 microbenchmark 拆出 commit wait 成本：“commit wait is about 5ms, and Paxos latency is about 9ms.”
  - “TrueTime 的前提可信” → §5.3 先提出两个问题：“is ε truly a bound on clock uncertainty, and how bad does ε get?” 并用机器统计回答：“bad CPUs are 6 times more likely than bad clocks.”
  - “真实应用受益” → §5.4 F1 案例：迁移理由、directory 分布、F1 视角延迟表。
  - 判断：评估围绕**洞察的前提**（ε 有界且小）组织，而不只是测系统吞吐。

### 2.5 Raft

- **元数据**：Diego Ongaro, John Ousterhout. “In Search of an Understandable Consensus Algorithm”. USENIX ATC 2014, pp. 305–319†. 链接：https://www.usenix.org/conference/atc14/technical-sessions/presentation/ongaro 。奖项：ATC'14 Best Paper（USENIX 论文页）。
- **一句话论点**：“Our approach was unusual in that our primary goal was understandability: could we define a consensus algorithm for practical systems and describe it in a way that is significantly easier to learn than Paxos?”（§1，p. 305）
- **引出问题的观察**：“Unfortunately, Paxos is quite difficult to understand, in spite of numerous attempts to make it more approachable. Furthermore, its architecture requires complex changes to support practical systems.”（§1）；§3 给出证据：“In an informal survey of attendees at NSDI 2012, we found few people who were comfortable with Paxos, even among seasoned researchers.” 以及作者自身：“a process that took almost a year.”
- **关键洞察**：可理解性可以作为设计准则被工程化：“we applied specific techniques to improve understandability, including decomposition (Raft separates leader election, log replication, and safety) and state space reduction”（§1）；§4 把它变为决策规则：“we evaluated the alternatives based on understandability”。
- **贡献类型**：新算法（与 Paxos 等价）+ 设计方法论。贡献以“novel features”列表呈现（Strong leader / Leader election / Membership changes），随后一段总括式主张：“We believe that Raft is superior to Paxos and other consensus algorithms, both for educational purposes and as a foundation for implementation.”（§1）
- **评估如何支撑论点**：主张被拆为可分别检验的几条（同段列出：simpler、complete、has implementations、proven、efficient），§8 一一对应：
  - “更易理解” → §8.1 43 名学生的交叉设计用户研究：“participants scored 4.9 points higher on the Raft quiz than on the Paxos quiz”，配 paired t-test 与回归，并在 Table 1 列出偏差来源与缓解措施。
  - “正确” → §8.2 TLA+ 形式化规约（约 400 行）与 safety 证明。
  - “效率相当” → §8.3 用论证（“the minimal number of messages”）+ leader election 收敛实验。
  - 判断：Raft 是“新目标需要新的评估方法”的极端例子——目标从性能换成可理解性，于是评估从 benchmark 换成用户研究。

### 2.6 Arrakis

- **元数据**：Simon Peter, Jialin Li, Irene Zhang, Dan R. K. Ports, Doug Woos, Arvind Krishnamurthy, Thomas Anderson, Timothy Roscoe. “Arrakis: The Operating System is the Control Plane”. OSDI 2014, pp. 1–16†. 链接：https://www.usenix.org/conference/osdi14/technical-sessions/presentation/peter 。奖项：OSDI'14 Best Paper。
- **一句话论点**：“We demonstrate that operating system protection is not contradictory with high performance.”（§1，p. 1）标题本身就是论点：OS 退为 control plane。
- **引出问题的观察**：硬件趋势 + 长期渐进优化失败：“Recent device hardware trends enable a new approach to the design of network server operating systems.”（Abstract）；“Much of this has been adopted in mainline commercial OSes, and yet it has been a losing battle: we show that the Linux network and file system stacks have latency and throughput many times worse than that achieved by the raw hardware.”（§1）；§2 量化：“Of the total 3.36 µs ... spent processing each packet in Linux, nearly 70% is spent in the network stack.”
- **关键洞察**：I/O 虚拟化硬件（SR-IOV 等）已能在硬件中做保护，所以 kernel 可以只做配置、不在每次操作的路径上：“Applications have direct access to virtualized I/O devices, allowing most I/O operations to skip the kernel entirely, while the kernel is re-engineered to provide network and disk protection without kernel mediation of every operation.”（Abstract）。同时设立约束排除廉价解：“it is easy to get good performance by extending the trusted computing base to include application code”（§1）。
- **贡献类型**：OS 架构（新抽象/分工）+ 原型 + 量化。编号贡献列表：“We make three specific contributions: • We give an architecture for the division of labor between the device hardware, kernel, and runtime ... (§3). • We implement a prototype ... (§3.8). • We use our prototype to quantify the potential benefits of user-level I/O for several widely used network services ... (§4).”（§1）——每条贡献附带章节号。
- **评估如何支撑论点**：§4 开头列出要回答的问题：“What are the major contributors to performance overhead in Arrakis and how do they compare to those of Linux (presented in §2)?” “Does Arrakis provide better latency and throughput for real-world cloud applications?” “Can Arrakis retain the benefits of user-level application execution and kernel enforcement, while providing high-performance packet-level network IO?” “What additional performance gains are possible by departing from the POSIX interface?” 第一问直接回指 §2 的开销分解；第三问对应“保护与性能不矛盾”的核心主张；第四问把“改 API”的收益与“不改 API”的收益分开报告。头条数字：Redis “2× better read latency, 5× better write latency, and 9× better write throughput compared to Linux.”（§1）

### 2.7 IX

- **元数据**：Adam Belay, George Prekas, Ana Klimovic, Samuel Grossman, Christos Kozyrakis, Edouard Bugnion. “IX: A Protected Dataplane Operating System for High Throughput and Low Latency”. OSDI 2014, pp. 49–65†. 链接：https://www.usenix.org/conference/osdi14/technical-sessions/presentation/belay 。奖项：OSDI'14 Best Paper。
- **一句话论点**：“We propose IX, an operating system designed to break the 4-way tradeoff between high throughput, low latency, strong protection, and resource efficiency.”（§1，p. 49）
- **引出问题的观察**：先陈述 conventional wisdom 再反驳：“The conventional wisdom is that aggressive networking requirements, such as high packet rates for small messages and microsecond-scale tail latency, are best addressed outside the kernel, in a user-level networking stack.”（Abstract）；随后指出现有方案的缺口：“While kernel bypass eliminates context switch overheads, on its own it does not eliminate the difficult tradeoffs between high packet rates and low latency (see §5.2). Moreover, user-level networking suffers from lack of protection.”（§1）
- **关键洞察**：借用 middlebox 的 control/data plane 分离 + 虚拟化硬件做保护 + run-to-completion 与自适应批处理同时优化带宽与延迟：“processing bounded batches of packets to completion ... This approach amortizes API overheads and improves both instruction and data locality. We set the batch size adaptively based on load.”（§1）
- **贡献类型**：OS 架构 + 一组设计原则（§3 列出 run to completion with adaptive batching 等）。无编号贡献列表，以末段总结两层意义：“IX demonstrates that, by revisiting networking APIs and taking advantage of modern NICs and multi-core chips, we can design systems that achieve high throughput and low latency and robust protection and resource efficiency. It also shows that, by separating the small subset of performance-critical I/O functions from the rest of the kernel, we can architect radically different I/O systems and achieve large performance gains, while retaining compatibility ...”（§1）
- **评估如何支撑论点**：基线同时选 Linux 与当时最强的用户态栈 mTCP：“IX outperforms Linux and mTCP by up to 10× and 1.9× respectively for throughput.” “The unloaded uni-directional latency for two IX servers is 5.7µs, which is 4× better than between standard Linux kernels and an order of magnitude better than mTCP, as both trade-off latency for throughput.”（§1）——用“对手在 latency/throughput 之间只能二选一”来证明“打破权衡”。应用层：memcached “up to 3.6× in terms of throughput at a given 99th percentile latency bound”，并给出原因测量：“reduce kernel time ... from ∼ 75% with Linux to < 10% with IX.” §5 小节（Latency and Single-flow Bandwidth / Throughput and Scalability / Connection Scalability / Memcached）与 4-way tradeoff 的各维度对应（判断）。
- **对比 Arrakis（判断）**：同一问题，Arrakis 框定为“OS 角色的重新划分”（标题即论点，强调应用与 POSIX 兼容/不兼容的收益分解），IX 框定为“打破一个四维权衡”（强调对 SOTA kernel-bypass 栈也有优势）。两者都以“保护不能放弃”作为排除廉价解的约束。

### 2.8 Shenango

- **元数据**：Amy Ousterhout, Joshua Fried, Jonathan Behrens, Adam Belay, Hari Balakrishnan. “Shenango: Achieving High CPU Efficiency for Latency-sensitive Datacenter Workloads”. NSDI 2019, pp. 361–378†. 链接：https://www.usenix.org/conference/nsdi19/presentation/ousterhout 。奖项：USENIX 页面无获奖标注。
- **一句话论点**：“Shenango achieves comparable latencies but at far greater CPU efficiency.”（Abstract）；“To our knowledge, Shenango is the first system that can both multiplex cores and maintain low tail latency during microsecond-scale bursts in load.”（§1）
- **引出问题的观察**：两条趋势相撞：“as Moore’s law slows and network rates rise, CPU efficiency becomes paramount”（§1）+ 负载在微秒尺度上突发：“micro bursts in Google’s Gmail servers cause sudden 50% increases in CPU usage, and, in Microsoft’s Bing service, 15 threads can become runnable in just 5 µs.”（§1）。然后提出“为什么”并定位根因：“Why do today’s systems force us to waste cores to maintain microsecond-scale latency? ... today’s schedulers only make thread balancing and core allocation decisions at coarse granularities (every four milliseconds for Linux and 50–100 milliseconds for Arachne and IX)”（§1，p. 361）。
- **关键洞察**：问题的根因是核心重分配的时间尺度，所以把重分配做到 5 µs 就能同时拿到低尾延迟与高效率：“Shenango solves the hard problem of reallocating cores across applications at very fine time scales; it reallocates cores every 5 microseconds, orders of magnitude faster than any system we are aware of.”（§1）
- **贡献类型**：新机制（核心分配算法 + IOKernel）。以“three goals”与“two key ideas”组织：“Shenango proposes two key ideas. First, ... an efficient algorithm that accurately determines when applications would benefit from additional cores ... Second, Shenango dedicates a single busy-spinning core per machine to a centralized software entity called the IOKernel”（§1）。并**显式定义指标**：“We define CPU efficiency as the fraction of cycles spent doing application-level work, as opposed to busy-spinning, context switching, packet processing, or other systems software overhead.”（§2）
- **评估如何支撑论点**：§2 “The Case Against Slow Core Allocators” 先用模拟证明慢分配器的上界不够（Figure 1：5 µs 的 Shenango 高于“an optimal simulation of a 1 ms core allocator”）——即在给出设计之前先证明“时间尺度”是根因。§7 列出三个问题：“1. How do latency and CPU efficiency compare for Shenango and other systems ...? 2. How well can Shenango respond to sudden bursts in load? 3. What is the contribution of the individual mechanisms in Shenango to its observed performance?” 分别对应：主张（与 ZygOS 同延迟但效率更高）、洞察（应对微秒突发）、机制必要性（消融）。

### 2.9 The Linux Scheduler: a Decade of Wasted Cores

- **元数据**：Jean-Pierre Lozi, Baptiste Lepers, Justin Funston, Fabien Gaud, Vivien Quéma, Alexandra Fedorova. “The Linux Scheduler: a Decade of Wasted Cores”. EuroSys 2016 (London). DOI 10.1145/2901318.2901326（取自 PDF 版权栏）。链接：https://people.ece.ubc.ca/sasha/papers/eurosys16-final29.pdf 。奖项：未查到†。
- **一句话论点**：“The main contribution of this work is the discovery and analysis of these bugs and providing the fixes.”（Abstract）
- **引出问题的观察**：先给一个简单不变量，再说它被破坏：“the OS thread scheduler must maintain the following, simple, invariant: make sure that ready threads are scheduled on available cores. As simple as it may seem, we found that this invariant is often broken in Linux. Cores may stay idle for seconds while ready threads are waiting in runqueues.”（Abstract）。时代背景：“Year 2004 brought an end to Dennard scaling, ushered in the multicore era ... These events once again made schedulers interesting, but at the same time increasingly more complicated and often broken.”（§1）。用 Linus 2001 年的引语 “scheduling is easy” 树立被推翻的共识。
- **关键洞察**：症状统一（work-conserving 不变量被长时间违反），所以一个在线不变量检查器就能捕获根因各异的一类 bug：“These bugs have different root causes, but a common symptom.”（§1）；且传统工具失效的原因被量化：“each time it lasted only a few hundreds of milliseconds – too short to detect with tools like htop, sar or perf.”（§1）
- **贡献类型**：刻画/发现（characterization）+ 修复 + 工具。贡献句见上；工具作为次要贡献：“we designed two new tools. The first tool, which we call a sanity checker, periodically checks for the violation of the aforementioned invariant ... The second tool visualizes traces of scheduling activity”（§1）。
- **评估如何支撑论点**：
  - “问题严重” → 修复前后对比：“Resulting performance degradations are in the range 13-24% for typical Linux workloads, and reach 138× in some corner cases.”（§1）；摘要中 TPC-H 下降为 14–23%（PDF 文本抽取丢失了连字符，§1 分别给出 14% 与 23%）。
  - “工具实用” → 可移植（“from Linux 3.17 through 4.3”）与开销“negligible”。
  - §5 Lessons Learned 把发现上升为架构主张：“What we need is to rethink the architecture of the scheduler, since it can no longer remain a small, compact and largely isolated part of the kernel.”
  - 补充：检索结果显示 Linux 内核邮件列表上有维护者对部分分析与补丁效果提出异议†（未逐字核实）。判断：刻画型论文的可信度依赖于结论能被他人复现，这类争议正说明“症状—根因—修复—度量”链条每一环都需要可检验。

### 2.10 Borg

- **元数据**：Abhishek Verma, Luis Pedrosa, Madhukar Korupolu, David Oppenheimer, Eric Tune, John Wilkes. “Large-scale cluster management at Google with Borg”. EuroSys 2015 (Bordeaux). DOI 10.1145/2741948.2741964（取自 PDF）。链接：https://research.google/pubs/large-scale-cluster-management-at-google-with-borg/ 。奖项：未查到†。
- **一句话论点**：“Borg is not the first system to address these issues, but it’s one of the few operating at this scale, with this degree of resiliency and completeness.”（§1）
- **引出问题的观察**：规模本身：“runs hundreds of thousands of jobs, from many thousands of different applications, across a number of clusters each with up to tens of thousands of machines.”（Abstract）
- **关键洞察**：prod 任务为罕见峰值预留但大多不用，所以 batch 可以跑在回收资源上，混部比分离省机器（§5.1 中给出原因，见下）。
- **贡献类型**：经验报告 + 设计决策的量化分析。摘要直接列出四类内容：“We present a summary of the Borg system architecture and features, important design decisions, a quantitative analysis of some of its policy decisions, and a qualitative examination of lessons learned from a decade of operational experience with it.” §8 分为 “Lessons learned: the bad” 与 “the good”。
- **评估如何支撑论点**：
  - **自定义评估指标并论证其合理性**：“to evaluate our policy choices we needed a more sophisticated metric than ‘average utilization’. After much experimentation we picked cell compaction”（§5.1）；并说明为何取 90 分位：“the mean or median would not reflect what a system administrator would do if they wanted to be reasonably sure that the workload would fit.” 以及指标的经济含义：“it translates directly into a cost/benefit result: better policies require fewer machines to run the same workload.”
  - “混部有价值” → “segregating prod and non-prod work would need 20–30% more machines in the median cell to run our workload. That’s because prod jobs usually reserve resources to handle rare workload spikes, but don’t use these resources most of the time.”（§5.1）
  - 判断：Borg 的主张是“规模与完整性”，因此评估的重点不是与他人比较，而是用反事实（“如果我们像别人一样分离部署会怎样”）量化每个策略决策。

---

## 3. 写作指南的标准

### 3.1 Levin & Redell, “How (and How Not) to Write a Good Systems Paper”

- **元数据**：Roy Levin, David D. Redell（第 9 届 SOSP 程序委员会联合主席）。原载 ACM SIGOPS Operating Systems Review 17(3), July 1983, pp. 35–40（据 USENIX 转载页注释）。链接：https://www.usenix.org/conferences/author-resources/how-and-how-not-write-good-systems-paper 。背景：83 篇投稿录用 16 篇，“the ten committee members quickly agreed on the disposition of over 80% of the papers.”
- **标准（按原文小节）**：
  - **Original Ideas**：“Can you state the new idea concisely? ... Try writing each idea down in a paragraph ... When you have the paragraphs, use them in the abstract for the paper.” “What exactly is the problem being solved? ... Be sure to explain why your problem couldn't be solved just as well by previously published techniques.” 警告：“the author of the paper sometimes unconsciously confuses the total effort with the work that is actually new.” 并指出真实系统论文最常见的问题是 “the systems described were frequently isomorphic to one of a small number of pioneering systems”。
  - **Reality**：“Does the paper describe something that has actually been implemented?” “If the system has been implemented, how has it been used, and what has this usage shown about the practical importance of the ideas?”
  - **Lessons**：“What have you learned from the work?” “How generally applicable are these lessons? Be sure to state clearly the assumptions on which your conclusions rest.”
  - **Choices**：“What were the alternatives considered at various points, and why were the choices made the way they were? A good paper doesn't just describe, it explains.” “Did the choices turn out to be right, and, if so, was it for the reasons that motivated them in the first place?”
  - **Context**：“What are the assumptions on which the work is based?” “Are they realistic?” “How sensitive is the work to perturbations of these assumptions?”
  - **Focus / Presentation**：摘要应是 “a precis of the technical content”，而不是 “a prose table of contents”；“include a simple statement of assumptions and results.”

### 3.2 Roscoe, “Writing reviews for systems conferences”

- **元数据**：Timothy Roscoe, ETH Zürich, March 2007（为 SOSP 2007 Shadow PC 所写的个人笔记，非正式出版）。链接：https://people.inf.ethz.ch/troscoe/pubs/review-writing.pdf 。
- **评审视角下的贡献标准（§4 Structure of a review）**：
  - 先复述再判断：“summarize the paper. Give a neutral description of what you think the paper is about, where the authors are coming from, why they view the problem as important, and what they’ve done.”
  - 贡献必须被明确陈述：“state what you think the contributions are. It’s rare to find a paper that completely fails to make any contributions, but it’s much more common to find one that makes no useful contributions, or contributions that turn out to be flawed.” “A surprisingly common mistake among paper authors is to fail to state the contribution”。
  - Novelty：“Does the related work invalidate the contribution, or (more likely) simply change it’s context or emphasis?”
  - 影响力：“Is the paper likely to prompt interesting discussion at the conference or workshop? Are the audience likely to take away some interesting ideas they can work with, beyond simply a feeling of good, solid, but otherwise uninspiring work?”
  - 成熟度与场合：“conference papers report on complete, mature work whereas workshop papers present early work or argue a position.”
  - 其他检查项：technical flaws、gaps or unaddressed issues、“anything you thought was really cool”。

### 3.3 Lampson, “Hints and Principles for Computer System Design”

- **元数据**：Butler Lampson. arXiv 2011.02455（PDF 版本日期 May 14, 2021），是其 1983 年 SOSP 论文 “Hints for Computer System Design” 的扩展版；1983 版获 SIGOPS Hall of Fame Award 2005（sigops.org）。1983 原版未能下载、未逐字阅读†。链接：https://arxiv.org/abs/2011.02455 。
- **性质**：这是**设计**准则而非论文评审准则；与本主题相关的是它对“好设计”的判断标准，可用来理解经典论文为何把某些洞察当作贡献（判断）。
- **相关标准**：
  - 系统设计的判断标准本身模糊：“The measure of success is much less clear.” “There probably isn’t a ‘best’ way to build it, or even any major part of it; what’s important is to avoid a terrible way, and to have clear division of responsibilities among the parts.”（§1，p. 3）
  - 最重要的提示：“Keep it simple. − Write a spec. − Design with independent modules. − Exploit the ABCs of efficiency. − Treat state as both being and becoming. − Use eventual consistency to keep data local.”（§1，p. 3）
  - 目标轴 STEADY：“Simple, Timely, Efficient, Adaptable, Dependable, Yummy”；技术轴 AID：“Approximate, Incremental, Divide & Conquer”。
  - 抽象是原则而非提示：“The most important ones are about abstraction and specs; those are principles, not just hints.”（§3.1.2）；“Approximate rather than exact, perfect or optimal results are almost always good enough, and often much easier and cheaper to achieve.”（§3.1.2）
  - **Inflection points**（§3.5.2 “Seize the moment. Ride the curve.”）：“The reverse happens when a new idea has some fundamental advantage that couldn’t be fully exploited in yesterday’s world, but conditions have changed so that it now pays off”，举例包括 VM 复兴、多核、日志结构文件系统因 SSD 物理特性重新变得有吸引力。

---

## 4. 归纳的模式

以下 7 个模式均为判断，由第 2 节卡片归纳。每个模式给出“论证骨架”、例证、以及它需要怎样的评估来闭环。

### 模式 1：环境或负载已变，旧假设失效 → 重新审视设计空间

- **骨架**：列出编号的观察（规模、故障率、访问模式、硬件能力、时间尺度）→ 每条观察否定一个传统假设 → 每个否定直接推出一个设计取舍。
- **例证**：
  - GFS：四条编号观察（“component failures are the norm rather than the exception” 等），结论再次复述 “We started by reexamining traditional file system assumptions”。
  - Arrakis：“Recent device hardware trends enable a new approach”，并论证渐进式内核优化是 “a losing battle”。
  - Shenango：负载在 µs 尺度突发，而调度器在 ms 尺度决策——时间尺度错配是根因。
  - Wasted Cores：Dennard scaling 结束 → 调度器复杂化 → 基本不变量失守。
- **评估闭环**：用生产数据或测量**回头验证前提**（GFS §6.3 Workload Breakdown；Arrakis §2 开销分解 “nearly 70% is spent in the network stack”；Shenango §2 模拟最优的 1 ms 分配器仍不够）。
- **对应短文标准**：Levin & Redell 的 Context（“What are the assumptions ... Are they realistic? How sensitive ...”）；Lampson 的 Inflection points。

### 模式 2：先陈述公认权衡或 conventional wisdom，再证明可以同时得到两端

- **骨架**：“通常认为 A 与 B 不可兼得 / 最好的做法是 X” → 指出现有方案只占据权衡的一端 → 新设计同时达到两端 → 评估用“对手各自只能赢一个维度”来证明。
- **例证**：
  - IX：“break the 4-way tradeoff between high throughput, low latency, strong protection, and resource efficiency”，对比 mTCP 时强调对手 “trade-off latency for throughput”。
  - Arrakis：“operating system protection is not contradictory with high performance”，并预先排除“扩大 TCB”这一廉价解。
  - Shenango：kernel-bypass 有低延迟但浪费 CPU，Linux 省 CPU 但延迟高；Shenango “achieves comparable latencies but at far greater CPU efficiency”。
  - Spanner：外部一致性与全球分布，“the first system to provide such guarantees at global scale”。
- **评估闭环**：基线必须包含**权衡两端各自最强的系统**（IX 同时比 Linux 与 mTCP；Shenango 比 ZygOS 与 Linux），且报告在固定 SLO 下的吞吐（IX：“at a given 99th percentile latency bound”）。
- **对应短文标准**：Levin & Redell 的 “explain why your problem couldn't be solved just as well by previously published techniques”；Roscoe 的 Novelty。

### 模式 3：一个关键事实使一个受限的抽象/接口足够，从而隐藏复杂度

- **骨架**：“X 在我们的负载中成立（无副作用、主键访问、时钟误差有界、append-mostly），所以可以提供更受限但更易实现的接口 Y，Y 让困难的问题（容错、一致性、并行化）变得简单。”
- **例证**：
  - MapReduce：计算是函数式的 → re-execution 即可容错；贡献明说是 “a simple and powerful interface”。
  - Spanner：时钟不确定性有界且可测 → TrueTime API 显式暴露不确定性，等待即得外部一致性；结论将其上升为 “reifying clock uncertainty in the time API”。
  - Dynamo：服务只需主键访问且可容忍分歧 → 版本化 + 应用协助合并，作者称之为 “a novel interface for developers to use”。
  - GFS：append-mostly → 放松一致性、加入 atomic record append。
- **评估闭环**：证明“受限接口仍足够表达真实工作负载”（MapReduce 的使用量与代码行数；Dynamo 的 99.94% 请求只见到一个版本），并测量“利用该事实的代价”（Spanner commit wait ≈ 5 ms）。
- **对应短文标准**：Lampson 把 abstraction/spec 称为 “principles, not just hints”，以及 “Approximate rather than exact ... almost always good enough”、“Use eventual consistency to keep data local”；Levin & Redell 的 “Can you state the new idea concisely?”——这类贡献最容易浓缩为一段话。

### 模式 4：重新定义目标或指标，并为新指标设计相应的评估方法

- **骨架**：现有评价维度不能反映真正的问题 → 明确定义新目标/新指标 → 用能测量该指标的方法评估（必要时发明方法）。
- **例证**：
  - Raft：目标从效率换成 “understandability”，于是评估用 43 名学生的交叉用户研究，并列出偏差缓解表。
  - Borg：“average utilization” 不足，提出 cell compaction，并论证为何取 90 分位、为何它 “translates directly into a cost/benefit result”。
  - Shenango：显式定义 “CPU efficiency as the fraction of cycles spent doing application-level work”。
  - Dynamo：SLA 以 99.9th 百分位衡量，而非均值或中位数。
- **评估闭环**：新指标需要**为其合理性单独辩护**（Borg 解释为何不用均值；Raft 列出偏差来源），否则评审会质疑指标被挑选来偏向自己。
- **对应短文标准**：Levin & Redell 的 Choices（“A good paper doesn't just describe, it explains”）；Lampson 的 “The measure of success is much less clear.”

### 模式 5：规模与生产经验本身作为证据，贡献以“组合 + 验证 + 教训”表述

- **骨架**：承认单项技术并不新 → 主张新知识在于“这些技术组合后在生产中的表现与调优” → 以长期运行数据和明确的教训作为证据。
- **例证**：
  - Dynamo：“The main contribution of this work for the research community is the evaluation of how different techniques can be combined to provide a single highly-available system.” 证据：99.9995% 成功率、分歧版本统计。
  - Borg：“Borg is not the first system to address these issues, but it’s one of the few operating at this scale”；“Lessons learned: the bad / the good”。
  - MapReduce / GFS：每日作业数、集群规模、生产索引系统重写；Spanner 的 F1 案例。
- **评估闭环**：生产数据回答“牺牲的东西实际有多小”（Dynamo）或“反事实下会差多少”（Borg 的分离部署需多 20–30% 机器）。
- **对应短文标准**：这一模式正面回应 Levin & Redell 对真实系统论文的主要批评（“isomorphic to one of a small number of pioneering systems”；“confuses the total effort with the work that is actually new”）——Dynamo 与 Borg 把新颖性主动放在“组合与经验”而非“单项机制”上；同时满足 Reality（“how has it been used”）与 Lessons（“How generally applicable are these lessons?”）。Roscoe 的 “contributions ... simply change it’s context or emphasis” 也对应此类定位。

### 模式 6：用一个简单不变量或理想上界做刻画，把“发现问题”本身作为贡献，或作为设计的前置证明

- **骨架**：写下系统**应当**满足的简单性质或可达到的上界 → 测量实际偏离多少 → 定位根因 → （可选）修复或设计新系统。
- **例证**：
  - Wasted Cores：不变量 “make sure that ready threads are scheduled on available cores”，发现被违反，贡献是 “the discovery and analysis of these bugs and providing the fixes”，并附带违反检测工具。
  - Arrakis §2：与 “raw hardware” 对比，逐阶段时间戳分解 Linux 的开销。
  - Shenango §2 “The Case Against Slow Core Allocators”：用模拟的最优 1 ms 分配器作上界，证明问题在时间尺度而非算法细节。
  - GFS §6 明言 micro-benchmarks 用于 “illustrate the bottlenecks inherent in the GFS architecture”。
- **评估闭环**：刻画需同时给出“偏离的量级”（Wasted Cores：13–24%，极端 138×）和“为何现有工具看不到”（持续时间 “too short to detect with tools like htop, sar or perf”）。
- **对应短文标准**：Levin & Redell 的 “Does the work comprise a significant extension, validation, or repudiation of earlier but unproven ideas?” 以及 Lessons；Roscoe 的 “prompt interesting discussion”。

### 模式 7：评估组织为“问题清单”，每个问题对应一条主张或一个机制

- **骨架**：引言里的主张被拆成若干可独立检验的子主张 → 评估开头列出编号问题 → 每个问题一个小节；包含至少一项消融（关闭某机制）或故障注入。
- **例证**：
  - Shenango §7：三问（对比、突发、各机制贡献），第三问即消融。
  - Arrakis §4：四问，第一问回指 §2 的开销分解，第四问把“改 POSIX API”的收益单独拿出。
  - Raft：引言的五条主张（simpler、complete、implemented、proven、efficient）对应 §8.1–8.3。
  - MapReduce：关闭 backup tasks（+44%）、杀死 200 个 worker（+5%）。
  - Spanner §5.3：先写出 “Two questions must be answered with respect to TrueTime” 再回答。
- **评估闭环**：问题清单让读者能检查“每条主张是否有对应证据”，消融证明“每个机制都必要”。
- **对应短文标准**：Roscoe 的 “state what you think the contributions are” 与 “Are there gaps or unaddressed issues?”——问题清单让评审几乎不用自己重建贡献；Levin & Redell 的 “Did the choices turn out to be right, and, if so, was it for the reasons that motivated them in the first place?”——消融正是对“为对的理由而正确”的检验。

### 4.8 短文标准与模式的映射总表

| 短文标准 | 主要对应模式 | 例证论文 |
|---|---|---|
| L&R Original Ideas：能用一段话说清新想法 | 3, 2 | MapReduce, Spanner, IX |
| L&R：说明为何已有技术不能同样好地解决 | 2, 1 | IX, Arrakis, Shenango |
| L&R Reality：已实现、如何被使用 | 5 | Dynamo, Borg, MapReduce |
| L&R Lessons：教训与适用范围、写明假设 | 5, 6 | Borg, Wasted Cores |
| L&R Choices：备选方案与选择理由；为对的理由而正确 | 4, 7 | Raft §4, Borg §5.1, MapReduce §5.4 |
| L&R Context：假设是否现实、对扰动是否敏感 | 1 | GFS, Spanner §5.3 |
| L&R：摘要是技术精要，含假设与结果 | 全部 | Arrakis/IX 摘要均含头条数字 |
| Roscoe：贡献必须被明确陈述 | 7 | Arrakis（编号贡献带章节号）, MapReduce |
| Roscoe：相关工作改变贡献的语境或侧重 | 5 | Dynamo, Borg |
| Roscoe：引发讨论、带走可用的想法 | 3, 6 | Spanner 结论对社区的主张, Wasted Cores |
| Lampson：抽象与 spec 是原则 | 3 | MapReduce, Spanner |
| Lampson：Approximate / eventual consistency | 3 | GFS, Dynamo |
| Lampson：Inflection points | 1 | Arrakis, Shenango, Wasted Cores |
| Lampson：Keep it simple / 成功标准模糊 | 4 | Raft, Borg |

### 4.9 跨模式的观察（判断）

- 10 篇中只有 Arrakis 使用编号贡献列表；MapReduce、Wasted Cores、Dynamo 用一句 “The main/major contribution(s) of this work is/are ...”；其余用论点句 + 结构化段落（GFS 的编号观察、Raft 的 novel features、Shenango 的 three goals / two key ideas）。显式的“contribution”句在不同年代都存在，但编号列表并不是经典论文的必要形式。
- 几乎每篇都在引言中给出**一个可被单句复述的洞察**，并以 “X, so Y” 的形式连接观察与设计；其余篇幅是这句洞察的实现与证据。
- 获奖/HoF 论文的评估很少只是“比基线快 N×”：要么验证洞察的前提（Spanner、GFS），要么量化放弃的东西（Dynamo），要么用反事实（Borg）或消融（MapReduce、Shenango）证明每个设计决定必要。
- 同一问题可有不同框定（Arrakis：角色重划；IX：打破权衡），两者同届获奖，说明框定方式本身不是决定因素，关键是框定与评估是否闭环。
