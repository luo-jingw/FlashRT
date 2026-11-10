# Characterizing VLA Models across XPUs（2604.24447）

- **VLA across XPUs** · 2026 · 预印本（comment："13 pages"）· [arXiv 2604.24447](https://arxiv.org/abs/2604.24447)
- Characterizing Vision-Language-Action Models across XPUs: Constraints and Acceleration for On-Robot Deployment
- 作者：Kaijun Zhou, Qiwei Chen, Da Peng, Zhiyang Li, Xijun Li, Jinyu Gu；v1 2026-04-27

## 论文做了什么

在 RTX 4090、Jetson AGX Orin、Jetson Thor、Intel B60、昇腾 310P 等硬件上刻画 VLA 推理，给出三部分：
- 模型–硬件配对的排行榜，按成本、能耗、时间（CET）评；
- 推理过程的刻画：VLM 主干受算力限制，动作专家受带宽限制，两段交替造成硬件利用率不均衡；
- 两个优化：DP-Cache（减少扩散步之间的冗余）与 V-AEFusion（VLM 与动作专家异步流水），GPU 上最多 2.9×、边缘 NPU 上最多 6×。

## 结构

1. Introduction
2. Related Work（VLA 模型、VLA 加速、异构硬件）
3. Model-Hardware Pairing（硬件分级、排行榜）
4. VLA Computation Characterization（两阶段不均衡、各阶段的计算特性）
5. Acceleration（正交优化的组合、DP-Cache、V-AEFusion）
6. Conclusion（含局限与后续工作）
- 附录：硬件规格、真机能耗、DP-Cache 补充评测

## Findings 的写法

全文 6 条，每条标成 "Finding #N"，先给一句结论，后面几段用数字和图表支撑：

1. **Finding #1**（§3.2）："Model size is not a definitive predictor of inference frequency."
   - 证据：图 2，各模型–硬件对的推理频率（对数轴），叠加 LIBERO 成绩，并标出 1–30 Hz 的控制频率线。
2. **Finding #2**（§3.2）：最优硬件随模型和指标而变。
   - 证据：图 3 能耗–延迟分解，图 4 CET 气泡图，表 1 π0 排行榜。
3. **Finding #3**（§4.1）："VLA workflow exhibits a distinct dual-phase computational imbalance, leading to severe resource underutilization."
   - 证据：图 5，π0 在 4090、Orin、Thor 上的 Nsight Systems 时间线与分阶段 SM 利用率（VLM 90% 以上，动作专家 20–40%）。
4. **Finding #4**（§4.2）：VLM 主干受算力限制，动作专家受带宽限制。
   - 证据：图 6，三个平台上的 roofline，配算术强度的计算（约 840 对 64.5 FLOPs/Byte）。
5. **Finding #5**（§5.2.2）："Phase-aware, orthogonal optimizations must be composed with a global view."
   - 证据：表 2、3、4、6。
6. **Finding #6**（§5.3.2）：同一优化在不同硬件上效果不同。
   - 证据：表 9，V-AEFusion 的加速在 4090 / Orin / Thor / B60 / 310P 上为 1.32× / 1.14× / 1.04× / 1.05× / 1.0×。

写法上的特点：
- 一条 Finding 只说一件事，多数只靠一张图；
- 图大多是单一指标，跨平台并列（Finding #3、#4、#6 都同时给了 Thor）；
- 刻画（第 3–4 节）先于优化（第 5 节），优化的动机直接引用前面的 Finding。

## 与本方向的关系

- Thor 上两阶段不均衡（Finding #3、#4）对应本方向"两个模型的阶段特性互补"的观察：规划器 prefill 吃算力、解码吃带宽，执行器在两者之间交替。
- Finding #6 中 V-AEFusion 在 Thor 上只有 1.04×：同一种并行在带宽受限的 SoC 上收益很小，与"SM 划分隔不住带宽"一致。
- 论文只刻画单个 VLA，没有讨论规划器与执行器两个模型共置。
