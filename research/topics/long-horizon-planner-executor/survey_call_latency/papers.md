# 论文卡片：多帧上下文下规划器单次调用的加速

快照 2026-11-19。本文件是 [brief.md](brief.md) 的文献部分：每篇论文压成一张卡片，对照我们的设置（Jetson AGX Thor，HF 官方 Qwen3.8-27B，64 层中 48 层 Gated DeltaNet + 16 层全注意力，batch = 1，1.2–1.8k token，近期 8 帧滑动 + 至多 8 张关键帧，约 1 Hz 调用，只需回复前几个 token）。
筛选：按 brief 的四类办法加"混合/线性注意力模型的缓存复用"一类，每类取 1–2 篇；优先 2025–2026 年直接对口的工作，其次是有影响力的奠基工作。每篇都在 arXiv 或会议页核对过，数字取自论文正文（标注表号/章节）。† 表示没能从论文正文核实。

| 论文 | 类别 | 重合度 | 一句话结论 |
|---|---|---|---|
| VisionZip（CVPR 2025） | 视觉 token 压缩 | 中 | ViT 输出里只有少数 token 有用，进 LLM 前压到约 1/9；省的是 prefill，不省 ViT |
| VLA-Cache（NeurIPS 2025） | 相邻帧 token 复用（机器人） | 中 | 相邻帧同位置的静态 patch 复用 KV，任务相关 patch 重算；前提是帧位置不变、全注意力 |
| StreamingVLM（ICLR 2026） | 流式视频 KV 复用 | 中 | sink + 近期视觉窗 + 文本窗，驱逐后把位置重排成连续；需 SFT，且全注意力 |
| CacheBlend（EuroSys 2025） | 非前缀 KV 复用 | 中 | 非前缀块的 KV 直接拼，只重算约 15 % 偏差最大的 token；作者明确不覆盖 Mamba 类 |
| VLCache（arXiv 2025.12） | 非前缀 KV 复用（多模态） | 高 | 同一图像出现在任意位置时跳过 ViT、复用 KV，只重算每张图开头 2–5 % token；只验证全注意力 Qwen3-VL |
| SGLang（NeurIPS 2024） | 输出侧：结构化输出 + 前缀缓存 | 中 | JSON 骨架里的常量文本一步跳过（compressed FSM）；对"只要前几个 token"直接有用但增量有限 |
| LinearKV（arXiv 2026.07） | 混合模型的位置无关缓存 | 高 | GDN 混合模型上，线性层只拿最后一块的缓存状态做初值 + 重算约 20 % token，恢复约 9 成质量；Qwen3.6-27B 在测试之列 |

---

### 1. VisionZip: Longer is Better but Not Necessary in Vision Language Models（CVPR 2025）· [arXiv 2412.04467](https://arxiv.org/abs/2412.04467)
- **一句话**：在视觉编码器内部按注意力挑出少数"主导 token"，其余按相似度合并成少量"上下文 token"，只把这几十个 token 送进 LLM。
- **intro 提出的问题 / 解决了哪些**：(1) 视觉 token 远多于文本 token（LLaVA-1.5 576 个，LLaVA-NeXT 2 880+）；(2) FastV、SparseVLM 在 LLM 内按文本–视觉注意力剪枝，多轮对话里保留的是上一个问题关心的 token。解决了 (1) 的 LLM 侧开销和 (2) 的文本依赖（选择与文本无关）；没解决 ViT 本身的开销——编码器仍处理全部 token，论文也没量化编码器部分。
- **方法：自然推导还是灵光一闪**：有一个非显然的观察——CLIP/SigLIP 的注意力集中在极少数 token 上，作者解释为编码器把信息"捷径式"汇聚到少数代理 token（§3，倒数第二层更集中）。有了这个观察，"直接按编码器注意力挑 token"就是顺推；观察本身是关键。
- **关键数字**：LLaVA-NeXT 7B 在 POPE 上 prefill 218 ms → 27.8 ms（7.8×，Table 4）；LLaVA-1.5 只留 64 token（−88.9 %）时，免训练版比 FastV 高 18.4 %、比 SparseVLM 高 8.2 %，微调投影层后保留 95.2 % 性能（Table 1）。
- **与我们的重合度**：中。直接缩短 prefill 长度，且与文本无关，可以跨调用缓存压缩后的帧；但实验都是 LLaVA 系（CLIP 类 ViT + 全注意力 LLM），没有 Qwen 系模型。
- **我们有、他们没考虑的**：我们一帧是两路 320×180 竖拼的小图，每帧 token 数本就不多，可压空间比 LLaVA-NeXT 的高分辨率小得多；规划器要分辨"子任务是否完成"这类细节，文本无关的压缩可能丢掉关键区域；最佳结果要微调投影层，而我们只用官方权重；压缩后 token 少了，48 层 GDN 的递归状态仍要逐 token 推进，节省比例与全注意力模型不同。

### 2. VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching（NeurIPS 2025）· [arXiv 2502.02175](https://arxiv.org/abs/2502.02175)
- **一句话**：闭环操作中相邻两帧同位置 patch 余弦相似度高的视为静态，复用上一步的 KV；任务相关的 patch 和注意力集中的层照常重算。
- **intro 提出的问题 / 解决了哪些**：(1) VLA 推理慢，影响实时控制；(2) 量化、早退等通用办法要改架构或重训；(3) 每步都重算大面积不变的背景。解决了 (3)，且免训练；但也指出朴素复用全部静态 token 会掉点（OpenVLA LIBERO-Spatial SR 84.4 % → 74.2 %，Table 1），靠"任务相关 token 剔除 + 逐层自适应比例"拉回 83.8 %。
- **方法：自然推导还是灵光一闪**：复用静态 patch 是顺着问题自然得到的；非显然的一步是"视觉上静止 ≠ 可以复用"——机械臂即将交互的区域看起来不变，但对动作关键，所以用文本→视觉注意力把它们从复用集里剔除（§3，Table 1 的消融正是这一步贡献了 SR）。
- **关键数字**：LIBERO 平均，OpenVLA 51.91 ms → 31.83 ms，SR 75.0 % → 74.7 %（Table 2）；真机 OpenVLA 64.16 ms → 51.85 ms，SR 82.1 % → 84.6 %（Table 5）。单卡 RTX 4090。
- **与我们的重合度**：中。场景最像我们（机器人、相邻帧大量相同），但它复用的是"同一位置的上一帧"，我们是"同一帧在下一次调用里换了位置"。
- **我们有、他们没考虑的**：复用依赖 patch 位置不变，静态 token 保留原位置编码（§3）；我们的帧在 prompt 里整体后移，需要位置修正。只测了 LLaMA2 底座（附录 A 自述），全部层都有逐 token KV；我们 48 层 GDN 没有可按 token 替换的 KV。VLA 一步只看当前帧，我们一次看 16 帧，复用对象是整帧而不是帧内 patch。附录 A 也承认动态场景下可复用 token 变少。

### 3. StreamingVLM: Real-Time Understanding for Infinite Video Streams（ICLR 2026）· [arXiv 2510.09608](https://arxiv.org/abs/2510.09608)
- **一句话**：推理时只保留 attention sink（512 token）+ 近 16 s 视觉 token + 近 512 文本 token 的 KV 并跨步复用，驱逐后把保留 token 的位置重排为连续；训练时用重叠短块全注意力 SFT 去匹配这个推理模式。
- **intro 提出的问题 / 解决了哪些**：(1) 全注意力随视频长度平方增长；(2) 无重叠滑窗丢连贯性、长块延迟高；(3) 有重叠滑窗每窗重算，延迟高；(4) 长视频训练代价大，已有 KV 驱逐与训练不对齐。四点都给出了办法；代价是必须做两阶段 SFT（论文称不 SFT 无法生成跨轮连贯内容，§2.2），且评测局限于英文体育解说。
- **方法：自然推导还是灵光一闪**：sink + 窗口是 StreamingLLM 的自然延伸；非显然的是两点——位置必须重排成连续，否则长流下崩（原生 RoPE 胜率 25.09 vs 连续 RoPE 66.18，Table 4）；以及"训练时重叠短块全注意力"就能近似推理时的 sink+窗口模式，不必在超长视频上训练（Table 7：重叠 66.18 vs 不重叠 62.51）。
- **关键数字**：Inf-Streams-Eval 对 GPT-4o mini 胜率 66.18（Table 1）；单张 H100 最高 8 FPS（§3.3）。论文没有给出重算滑窗基线的逐帧延迟数值。
- **与我们的重合度**：中。它正面处理了"窗口滑动后已缓存 KV 的位置怎么办"，并且是 Qwen 系（Qwen2.5-VL-7B，3D mRoPE 版的连续位置）；但它的上下文是单调追加 + 从头驱逐，我们的 prompt 每次重排（关键帧插在近期帧之前）。
- **我们有、他们没考虑的**：需要微调，我们只用官方权重；连续位置重排在全注意力层里可以通过旋转已缓存 K 实现†（论文未写清缓存 K 的重旋转细节），在 GDN 层无对应操作——GDN 状态里没有位置可改，"驱逐最老帧"在递归状态上无法执行；它连续输出解说文本，我们每次是独立的短决策调用。

### 4. CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion（EuroSys 2025）· [arXiv 2405.16444](https://arxiv.org/abs/2405.16444)
- **一句话**：各块独立预计算 KV，使用时直接拼接（不限前缀），每层只重算 KV 偏差最大的约 15 % token 来补回块间交叉注意力，并把重算与下一层 KV 加载流水化。
- **intro 提出的问题 / 解决了哪些**：(1) 前缀缓存只复用开头那一块，多块输入时接近全量重算；(2) 非前缀块直接复用丢了块间交叉注意力，答案出错，块越多越差（§3.3，Fig. 2–3）。两点都解决：质量与全量重算差约 0.02 F1/Rouge-L（§7.2）。作者自述只适用于 Transformer，Mamba、Griffin 留作未来工作（§9）。
- **方法：自然推导还是灵光一闪**：有两个非显然观察。一是只有约 10–15 % 的 token KV 偏差显著（注意力稀疏，§4.3，Fig. 7）；二是偏差大的 token 在相邻层基本一致（Fig. 8），所以可以在第一层多挑一点、逐层收窄，而无需先算出真实偏差。第二个观察是让方法可行的关键。RoPE 的位置修正（对缓存 K 乘旋转矩阵，附录 A）是自然推导。
- **关键数字**：TTFT 比全量重算低 2.2–3.3×、吞吐高 2.8–5×（§7.2，Fig. 12/14）。模型 Mistral-7B / Yi-34B / Llama-70B，A40。
- **与我们的重合度**：中。"块可出现在任意位置"正对应我们关键帧与近期帧位置每次变化的问题；但它是文本 RAG，块之间语义独立，且重点在把 KV 放在慢存储上的流水化。
- **我们有、他们没考虑的**：48 层 GDN 没有逐 token KV，"选 15 % token 重算"在这些层无从下手（见卡片 7 LinearKV）；我们 batch = 1、1–2k token、统一内存，流水化"从 SSD 加载 KV 与重算重叠"这一卖点不适用，而重算本身在 Thor 上与执行器争带宽；我们的帧之间高度相关（同一场景相邻 0.5 s），块间交叉注意力的偏差分布可能与 RAG 不同。

### 5. VLCache: Computing 2% Vision Tokens and Reusing 98% for Vision-Language Inference（arXiv 2025.12，Qwen 团队）· [arXiv 2512.12977](https://arxiv.org/abs/2512.12977)
- **一句话**：按图像哈希缓存 ViT 输出与 RoPE 之前的 KV；同一张图出现在任意位置时跳过 ViT、复用 KV，只重算每张图开头 r 比例的 token（平均 r 约 2–5 %），各层重算比例按离线测得的敏感度分配。
- **intro 提出的问题 / 解决了哪些**：(1) CacheBlend、KVShare 的局部偏差度量选不准该重算的 token；(2) MPIC 依赖 attention sink，作者认为复用的视觉 token 上不出现这种效应。给出了替代的选择规则；没有覆盖"相似但不相同的图"（结论里列为未来工作），没讨论视频。
- **方法：自然推导还是灵光一闪**：关键观察是复用误差会沿序列向后累积（§3.1），所以重算每张图最前面的 token 比挑"偏差大"的 token 更有效；再加上层间敏感度差异很大（§3.2）。前者是对 CacheBlend 思路的一个非显然修正；层间分配是自然推导。缓存 RoPE 之前的 K 让"换位置"免费，这一点简单但对我们最关键。
- **关键数字**：Qwen3-VL-8B 平均重算 3.5 % 时精度 74.49 vs 全量 74.48（Table 3）；TTFT 在 20K token 下 Qwen3-VL-8B 约 1.86×、Qwen3-VL-32B 约 3.18×（Table 1/2）。H20。
- **与我们的重合度**：高。同为 Qwen 系 VLM、同一图像在不同位置复用、连 ViT 一起跳过，几乎就是"上一调用的帧在本调用里换了位置"的问题。
- **我们有、他们没考虑的**：只测全注意力 Qwen3-VL，全文没有提线性注意力或混合模型——我们 48 层 GDN 的状态不能按图复用；TTFT 收益在 20K token 上测得，1–2k token 时固定开销占比更大，收益未知；只复用完全相同的图（哈希命中），我们的帧是逐帧新图、只在多次调用之间重复出现，这恰好满足哈希命中条件；只测单请求，没有共置执行器的带宽竞争。

### 6. SGLang: Efficient Execution of Structured Language Model Programs（NeurIPS 2024）· [arXiv 2312.07104](https://arxiv.org/abs/2312.07104)
- **一句话**：运行时用 RadixAttention（基数树 + LRU 的前缀 KV 缓存）跨调用复用前缀，用 compressed FSM 在约束解码里把只有唯一合法延续的常量文本一步吐出。
- **intro 提出的问题 / 解决了哪些**：(1) 多次调用共享前缀但 KV 不复用；(2) 约束解码每步只出一个 token。两点都解决；前缀复用只限严格前缀，非前缀不管。
- **方法：自然推导还是灵光一闪**：compressed FSM 的观察朴素但有用——`{"summary": ` 这类常量文本跨多个 token 却只有一种合法延续，可以合并成一步（§4）；需要对压缩边界重新分词以免切错 token。RadixAttention 是前缀缓存的工程化，属自然推导。
- **关键数字**：JSON 解码基准上 compressed FSM 吞吐 1.6×（§6）；单程序不 batch 的时延最多降 3.7×（§6.1，Llama-7B）。
- **与我们的重合度**：中。我们的回复就是 JSON，且只要其中一个字段；跳过骨架 token + "决策字段闭合即停"可以直接压缩解码段。
- **我们有、他们没考虑的**：骨架 token 少（我们回复 10–18 token 时，跳过骨架最多省几个 token × 24–36 ms）；我们已有 MTP 投机解码，骨架这类确定性 token 很可能已被 MTP 高接受率覆盖，两者叠加的增量未知；RadixAttention 对 GDN 层需要在前缀边界存递归状态检查点（见卡片 7 与"考虑过但未收"中的 Marconi）；没有讨论"只需第一个字段"时提前终止。

### 7. LinearKV: One Cached State Suffices for Position-Independent Caching in Hybrid LLMs（arXiv 2026.07）· [arXiv 2608.11231](https://arxiv.org/abs/2608.11231)
- **一句话**：把位置无关缓存（PIC）搬到混合模型：全注意力层照常拼 KV；每个线性层只取一块（默认最后一块）的缓存状态作初值，再沿用 CacheBlend/EPIC 等选出的约 20 % token 依序重算。
- **intro 提出的问题 / 解决了哪些**：(C1) 混合模型的线性层只有固定大小状态，没有逐 token KV 可拼、可选择性修补，现有 PIC 能否复用；(C2) 不同递归族（Mamba-2 vs GDN）和不同比例下能否用同一做法。对 GDN 两个模型给出了可用答案；Mamba-2 只测了一个模型，边界是经验性的（§3.2 自述）。
- **方法：自然推导还是灵光一闪**：自然的想法是把各块状态精确组合成完整前缀状态（同期 HYPIC 的做法）；非显然的发现是"只用一块的状态"在 GDN 上与精确组合打平，在 Mamba-2 上反而好得多——精确组合的误差逐层累积（Granite 第 32 层达到约 2 倍状态范数，§3.2，Fig. 3），GDN 的门控稠密更新能压住这种误差。这是观察驱动的，不是推导出来的。
- **关键数字**：Granite（Mamba-2）+ EPIC，LongBench QA 质量恢复 46.6 %（精确组合）vs 86.8 %（单状态）（Table 1）；GDN 模型两者差 ≤ 0.013 F1，最好恢复约 92 %（Table 2）；32K 时 TTFT 为全量 prefill 的 0.46–0.62×（Table 3，单张 H800，batch 1）。
- **与我们的重合度**：高。测试模型中包括 Qwen3.6-27B（16 层全注意力 + 48 层 GDN，和我们的层结构相同），问题正是"非前缀复用在线性层上怎么做"。
- **我们有、他们没考虑的**：只测文本（LongBench、RULER），最短 8K，没有 1–2k、没有多模态；块固定为 512 token，而我们的自然单元是"一帧"；作者自述批量服务和 host→device 传输未研究，统一内存下没有传输但有带宽竞争；20 % 重算在 1–2k token 下省下的绝对时间小，GDN 层仍需按序推进所选 token，Thor 上的实际收益要实测；视觉 token 相邻帧高度相似，"上一块的状态"可能比文本场景更接近真值†（推测，论文无多模态数据）。

---

## 综合

- **判读表第 1 行（图像编码 + prefill 占一半以上）对应 VLCache 和 LinearKV。** VLCache 解决"同一帧换位置后 ViT + 全注意力 KV 复用"，LinearKV 解决"线性层没有逐 token KV 时怎么做非前缀复用"，两者合起来正好覆盖我们 16 + 48 层的两半，但还没有人把它们放在同一个多模态、短上下文的系统里验证过。VisionZip、VLA-Cache 是第二梯队：前者缩短序列、不省 ViT，后者依赖帧位置不变。
- **判读表第 2 行（解码占一半以上）对应 SGLang 的 compressed FSM + 我们已有的"决策闭合即停"。** 文献里没有专门针对"只要第一个字段"的工作；增量主要来自减少要生成的 token 数，而不是加速每个 token（MTP 已在位）。
- **判读表第 4 行（文献办法依赖全注意力 KV 可拼接性）基本被 2026 年的混合模型缓存工作回答了一半**：LinearKV（位置无关）、Tail-Replay / SuffixReplay（前缀缓存、尾部重放）都利用了 GDN 的门控遗忘，用近期少量 token 重建状态。空白在于它们都是长文本（≥ 8K）、H800 级 GPU、文本任务。
- **未覆盖的空白 1：短上下文 + 带宽受限下，部分复用是否还划算。** 所有论文的收益都在 8K–32K token 上测得；我们 1.2–1.8k token、Thor 273 GB/s 统一内存，27B 权重的读取量随重算 token 数变化不大，prefill 可能接近权重带宽下限而不是算力下限，"只算 20 % token"能省多少要实测。
- **未覆盖的空白 2：prompt 布局本身是一个变量。** 记忆关键帧插在近期帧之前，导致每次调用从记忆区之后全部失配；把新帧放在末尾、记忆区只在关键帧变化时更新，就能把大部分输入变成严格前缀（GDN 只需在边界存一个状态检查点），但这会改变 MemER 提示格式、可能影响决策质量。没有论文研究"为可缓存性重排多帧 prompt"对 VLM 决策质量的影响。另外，与周期执行器共享带宽时，部分复用带来的 prefill 节省如何转化为执行器的时延余量，也没有论文涉及。

## 考虑过但未收

- FastV（ECCV 2024，[2403.06764](https://arxiv.org/abs/2403.06764)）：观察有洞见（第 2 层后视觉 token 注意力很低），但在 LLM 内部按文本剪枝，和 VisionZip 同类且被其超过；层内剪 token 与 GDN 递归状态的交互不明。
- SparseVLM：同属文本引导的 LLM 内剪枝，VisionZip 已与其比较（Table 1/4）。
- ReKV（ICLR 2025，[2503.00540](https://arxiv.org/abs/2503.00540)）：面向长视频 + 检索式 KV 回取，假设先编码后提问、视频很长；我们只有 16 帧，检索没有意义。StreamingVLM 报告 ReKV 与其固定上下文格式不兼容（Table 2）。
- LiveVLM（[2505.15269](https://arxiv.org/abs/2505.15269)）、StreamMem（[2508.15717](https://arxiv.org/abs/2508.15717)）：流式视频 KV 压缩，目标是长视频内存有界，与我们短窗口、低时延的需求错位。
- MPIC（[2502.01960](https://arxiv.org/abs/2502.01960)）：多模态位置无关缓存，思路被 VLCache 覆盖且 VLCache 指出其 attention sink 假设在复用视觉 token 上不成立。
- Marconi（MLSys 2025，[2411.19379](https://arxiv.org/abs/2411.19379)）：首个混合模型前缀缓存系统（状态检查点 + 准入/驱逐策略），面向多请求服务的缓存管理；我们是单请求、严格前缀，检查点只需一个，策略部分用不上。
- Tail-Replay（[2608.30310](https://arxiv.org/abs/2608.30310)）、SuffixReplay（[2609.33477](https://arxiv.org/abs/2609.33477)）：与 LinearKV 同组作者，解决的是混合模型的**前缀**缓存（任意边界命中，不存检查点）；我们的问题是非前缀，LinearKV 更对口。若改成"新帧追加在末尾"的布局，这两篇转为直接相关（SuffixReplay 测了 Qwen3.6-27B-FP8）。
- HYPIC（[2607.01299](https://arxiv.org/abs/2607.01299)）与 NeurIPS 2026 海报 "Bridging the Gap: Position-Independent Cache Reuse for Hybrid SSM-Attention Architectures"†：同为混合模型的位置无关缓存；HYPIC 是 LinearKV 对照的"精确组合各块状态"做法，LinearKV 已与之比较，故只收 LinearKV。海报正文未读。
- EAGLE-3 等投机解码：我们已用 MTP，同属一类；回复只有 10–18 token 时，换投机方法的增量预计小于减少输出 token 数。
