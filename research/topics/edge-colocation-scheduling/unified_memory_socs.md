# 统一内存 SoC / APU 硬件规格对照（CPU 与 GPU 共享同一物理 DRAM）

用途：比较大 LLM planner 与小型实时 action model 共享同一内存池时的平台差异。基线为 NVIDIA Jetson AGX Thor (T5000)。

检索日期：2026-11-17。

## 标注约定

- 无标记的数字：来自厂商官方页面或 datasheet（nvidia.com / developer.nvidia.com / docs.nvidia.com、apple.com / support.apple.com、amd.com / instinct.docs.amd.com、qualcomm.com、intel.com）。
- `†`：官方资料中未找到，取自可信二手来源或 NVIDIA 开发者论坛中的用户实测，来源见「来源链接」列。
- `推算`：由官方数字计算得出，计算方式写在单元格内，不是厂商声明。
- `未公开`：本次检索的官方资料中没有该数字。
- 算力一律按厂商原文标注精度及 sparse / dense。NVIDIA 的 headline 数字（Thor 2070、DGX Spark 1 PFLOP、Orin 275 TOPS）都是 **sparse**；AMD Ryzen AI Max 的 60 TFLOPS FP16 与 Qualcomm 的 4.6 TFLOPS 未注明 sparse / dense；Apple 不公布 GPU TFLOPS。
- 来源编号 [Sn] 对应文末「检索记录」。

## 主表

| 设备 | 发布年份 | GPU 架构 | GPU 计算单元数 | GPU 峰值算力（厂商口径） | CPU 核数与架构 | 内存类型与容量（最大配置） | 内存带宽 GB/s | 功耗范围 | MIG / 分区或优先级机制 | 来源链接 |
|---|---|---|---|---|---|---|---|---|---|---|
| **NVIDIA Jetson AGX Thor (T5000)** — 基线 | 2025（2025-08-25 GA） | Blackwell（CC 11.0 / sm_110 †） | 2560 CUDA cores，10 TPC；SM 未公开（用户实测 20 SM †）；Tensor Core 数官方 blog 写 96，NVIDIA 论坛称今后不再公布 | FP4: 2070 TFLOPS sparse / 1035 dense；FP8: 1035 sparse / 517 dense；FP16: 517 sparse；INT8: 1035 sparse | 14 核 Arm Neoverse-V3AE，2.6 GHz | 128 GB 256-bit LPDDR5X | 273 | 40–130 W | MIG（官方：MIG with 10 TPCs）。JetPack 7.2 实测 profile：1g=6 SM、2g=12 SM、3g=20 SM，profile 名为 `Ng.0gb`（不切分内存容量）† | [S1][S2][S3][S4] |
| NVIDIA Jetson T4000 | 2026（2026-01-07 JetPack 7.1 支持发布） | Blackwell | 1536 CUDA cores，6 TPC；Tensor Core 64（blog，标注 preliminary） | FP4: 1200 TFLOPS sparse / 600 dense；FP8: 600 sparse / 300 dense | 12 核 Arm Neoverse-V3AE | 64 GB 256-bit LPDDR5X | 273 | 40–70 W | MIG with 6 TPCs | [S1][S2][S5] |
| NVIDIA Jetson AGX Orin 64GB | 2022（dev kit 2022-03） | Ampere | 2048 CUDA cores、64 Tensor Cores；推算 16 SM（2048÷128） | INT8: 275 TOPS sparse（整模块含 DLA）；GPU Tensor INT8 170 sparse / 85 dense；GPU Tensor FP16 85 sparse / 43 dense；CUDA core FP32 5.3 TFLOPS | 12 核 Arm Cortex-A78AE v8.2，2.2 GHz | 64 GB 256-bit LPDDR5 | 204.8 | 15–60 W | 无 MIG（规格表未列）；另有 2× NVDLA v2 可分担推理 | [S6][S7] |
| NVIDIA Jetson AGX Orin 32GB | 2022 | Ampere | 1792 CUDA cores、56 Tensor Cores | INT8: 200 TOPS sparse（同页另一表写 241，官方页自相矛盾）；GPU Tensor INT8 108 sparse / 54 dense；FP16 54 sparse / 27 dense | 8 核 Cortex-A78AE，2.2 GHz | 32 GB 256-bit LPDDR5 | 204.8 | 15–60 W | 无 MIG；1× NVDLA v2 | [S6] |
| NVIDIA Jetson Orin NX 16GB (Super) | 2022 宣布；Super 模式 2024-12 | Ampere | 1024 CUDA cores、32 Tensor Cores | INT8: 157 TOPS sparse（Super）；GPU Tensor INT8 77 sparse / 38 dense；FP16 38 sparse / 19 dense | 8 核 Cortex-A78AE，2 GHz | 16 GB 128-bit LPDDR5 | 102.4 | 10–40 W | 无 MIG | [S6][S8] |
| NVIDIA Jetson Orin Nano Super Dev Kit 8GB | 2024（2024-12-17） | Ampere | 1024 CUDA cores、32 Tensor Cores | INT8: 67 TOPS sparse / 33 dense；FP16 33 sparse / 17 dense | 6 核 Cortex-A78AE，1.7 GHz | 8 GB 128-bit LPDDR5 | 102 | 7–25 W | 无 MIG | [S6][S8] |
| NVIDIA DGX Spark (GB10 Grace Blackwell) | 2025（2025-10-15 开售 †） | Blackwell（sm_121 †） | 6144 CUDA cores；SM 未公开（推算 48 SM = 6144÷128） | FP4: up to 1 PFLOP **sparse**（页面脚注 "using the sparsity feature"）；dense 未公开 | 20 核 Arm：10× Cortex-X925 + 10× Cortex-A725 | 128 GB 256-bit LPDDR5X（OEM 另有 64 GB），"coherent unified system memory" | 273 | GB10 TDP 140 W；整机电源 240 W | 官方文档未提及 MIG / 分区 | [S9][S10][S11] |
| Apple M4 | 2024（2024-05-07） | Apple GPU | 10 Apple GPU cores | 未公开（Neural Engine 38 TOPS，非 GPU） | 10 核：4P + 6E | 最大 32 GB unified memory | 120 | 芯片功耗未公开（Mac mini 2024 整机最大持续 155 W） | 未公开 | [S12][S13] |
| Apple M4 Pro | 2024（2024-10-30） | Apple GPU | 最多 20 Apple GPU cores | 未公开 | 最多 14 核：10P + 4E | 最大 64 GB unified memory | 273 | 芯片功耗未公开（Mac mini 整机 155 W） | 未公开 | [S13][S14] |
| **Apple M4 Max** | 2024（2024-10-30） | Apple GPU | 最多 40 Apple GPU cores（32-core 版为 410 GB/s） | 未公开 | 最多 16 核：12P + 4E | 最大 128 GB unified memory（仅 16 核 CPU / 40 核 GPU 版） | 546 | 芯片功耗未公开（Mac Studio 2025 整机最大持续 480 W） | 未公开 | [S14][S15] |
| Apple M3 Ultra | 2025（2025-03-05） | Apple GPU | 最多 80 Apple GPU cores | 未公开 | 最多 32 核：24P + 8E | 发布时最大 512 GB；当前 support 页可选 96 / 256 GB | 819（新闻稿 "over 800"） | 芯片功耗未公开（Mac Studio 整机 480 W） | 未公开 | [S15][S16] |
| Apple M5 | 2025（2025-10-15） | Apple GPU，每个 GPU core 带 Neural Accelerator | 10 Apple GPU cores | 未公开 TFLOPS；官方称 "over 4x peak GPU compute (for AI) vs M4" | 10 核：4P + 6E | 新闻稿提及 32 GB | 153 | 未公开 | 未公开 | [S17] |
| Apple M5 Max | 2026（2026-03-03） | Apple GPU + Neural Accelerators；Fusion Architecture（双 die） | 最多 40 Apple GPU cores | 未公开 TFLOPS；"over 4x peak GPU compute for AI vs previous generation" | 18 核：6 super + 12 performance | 最大 128 GB | 614（32-core GPU 版 460） | 未公开 | 未公开 | [S18][S19] |
| Apple M5 Ultra | 2026（2026-08-25 发布，09-22 到货） | Apple GPU + Neural Accelerators | 最多 80 Apple GPU cores | 未公开 TFLOPS；"up to 4.3x peak AI compute vs M3 Ultra" | 最多 36 核：12 super + 24 performance | 最大 512 GB | 1200（"1.2TB/s"） | 未公开 | 未公开 | [S19][S20] |
| Apple M6 | 2026（2026-08-25） | Apple GPU + Neural Accelerators | 12 Apple GPU cores | 未公开 TFLOPS | 12 核：2 super + 4P + 6E | 最大 32 GB | 170（16 GB 配置 153） | 未公开 | 未公开 | [S19][S21] |
| **AMD Ryzen AI Max+ 395 (Strix Halo)** | 2025（2025-01-06 CES 发布，Q1 2025 上市） | RDNA 3.5（Radeon 8060S，gfx1151 †） | 40 CU，2.9 GHz | FP16 AI Compute: up to 60 TFLOPS（sparse/dense 未注明）；整芯片 126 TOPS，NPU 50 TOPS | 16 核 / 32 线程 Zen 5，最高 5.1 GHz | 最大 128 GB 256-bit LPDDR5x-8000；"up to 96GB available for graphics" | 256（Halo 开发平台页） | 默认 TDP 55 W；cTDP 45–120 W | 未公开；GPU 可用容量由 BIOS / VGM 配置（官方 blog：可配到 64 GB 以上） | [S22][S23][S24][S25] |
| AMD Instinct MI300A | 2023（2023-12-06） | CDNA 3 | 228 CU、14,592 stream processors | FP8: 1.96 PFLOPs dense / 3.92 sparse；FP16: 980.6 TFLOPs dense / 1.96 PFLOPs sparse；FP64: 61.3 TFLOPs（matrix 122.6） | 24 核 Zen 4，3.7 GHz | 128 GB HBM3（8192-bit），CPU 与 GPU 共用 | 5300 | TDP 550 W，峰值 760 W | 计算分区 SPX（1×228 CU）/ TPX（3×76 CU）/ CPX（6×38 CU）；内存仅 NPS1；文档明确分区间无内存隔离 | [S26][S27] |
| Qualcomm Snapdragon X Elite (X1E-84-100) | 未核实 | Adreno | 未公开 | up to 4.6 TFLOPS（精度未注明）；NPU 45 TOPS | 12 核 Qualcomm Oryon，3.8 GHz 全核 / 4.2 GHz 双核 boost | 最大 64 GB LPDDR5x-8448 | 135 | 未公开 | 未公开 | [S28] |
| Qualcomm Snapdragon X2 Elite Extreme (X2E-96-100) † | 2025（2025-09 发布 †） | Adreno X2-90 † | 未公开 | 未公开（Qualcomm 不再公布 TFLOPS †） | 最多 18 核 Oryon † | 128 GB LPDDR5X-9523，192-bit † | 228 † | 未公开 | 未公开 | [S29][S30] |
| Qualcomm Snapdragon 8 Elite (SM8750-AB) | 2024（2024-10 †） | Adreno（sliced architecture） | 未公开 | 未公开 | 8 核 Oryon：2 Prime 4.32 GHz + 6 Performance 3.53 GHz（核数 †；官方 brief 只给频率） | 最大 24 GB dual-channel LPDDR5x，5.3 GHz | 未公开 | 未公开 | 未公开 | [S31][S32] |
| Intel Core Ultra 9 288V (Lunar Lake) | 2024（Q3'24） | Xe2（Arc 140V） | 8 Xe-cores，2.05 GHz | GPU INT8 67 TOPS；NPU 48 TOPS；整芯片 120 TOPS | 8 核：4P + 4E，TSMC N3B | 最大 32 GB on-package LPDDR5X-8533 | 未公开；推算 136.5（128-bit † × 8533 MT/s） | 基础 30 W，最大 turbo 37 W，最低保证 17 W | 未公开 | [S33][S34] |
| *对照：NVIDIA GH200（非统一内存）* | 2023（2023-05-28 量产） | Hopper GPU + Grace CPU，NVLink-C2C 相连 | 未在本次检索的 GH200 文档中列出 | FP8: 3958 TFLOPS sparse / 1979 dense；FP16/BF16: 1979 sparse / 990 dense；FP64 34 | 72 核 Arm Neoverse V2 | 两块独立物理内存：CPU 侧最大 480 GB LPDDR5X + GPU 侧 96 GB HBM3 或 144 GB HBM3e（合计最多 624 GB） | LPDDR5X 最高 512（另一版 384）；HBM 4000 / 4900；C2C 900 | 可配置 450–1000 W（内存+CPU+GPU） | GPU 侧支持 Secure MIG | [S35][S36][S37] |

### GH200 为何不算统一内存

- 物理上是两块内存：Grace 侧 LPDDR5X 与 Hopper 侧 HBM3/HBM3e，各自带宽不同（≤512 GB/s 对 4–4.9 TB/s），由 900 GB/s 的 NVLink-C2C 连接 [S35]。
- 编程模型是 coherent 而非 unified：ATS 让 CPU 与 GPU 共用一张 per-process page table，NVLink-C2C 支持 cache-line 粒度的互访；但内存在 first touch 时落到 LPDDR5X 或 HBM 之一，"From an OS perspective, the Grace CPU and Hopper GPU are just two separate NUMA nodes"，runtime 还可以迁移页面 [S36]。
- 后果：GPU 访问 CPU 侧内存走 C2C（900 GB/s 总带宽），访问 HBM 走本地 4.9 TB/s，访问成本取决于页面位置；而 Thor / Spark / Orin / Apple / Strix Halo / MI300A 的 CPU 与 GPU 访问的是同一组 DRAM 通道。

### 已排除或未加入的设备

- 无设备因"不共享内存"被剔除：上述 SoC 的 GPU 都是集成 GPU，使用系统 DRAM。
- Intel Panther Lake、AMD 后续 APU：本次未检索，未加入。

## 与 Thor 的关键差异

- **带宽档位**：Thor T5000 / T4000、DGX Spark、M4 Pro 都是 273 GB/s（前两者同为 256-bit LPDDR5X）；Strix Halo 256、AGX Orin 204.8、Orin NX / Nano ≈102；高于 Thor 的有 M4 Max 546、M5 Max 614、M3 Ultra 819、M5 Ultra 1200、MI300A 5300。
- **容量档位**：128 GB 是 Thor T5000、DGX Spark、Ryzen AI Max+ 395、M4 Max、M5 Max、MI300A 的共同上限；M3 Ultra（发布时）与 M5 Ultra 到 512 GB；T4000 与 AGX Orin 64GB 为 64 GB。Strix Halo 官方写明其中最多 96 GB 可分给 graphics。
- **算力 / 带宽比**：Thor T5000 dense FP8 517 TFLOPS 对 273 GB/s；DGX Spark 只公布 FP4 sparse 1 PFLOP（Thor 为 2070 sparse），带宽相同；Strix Halo 公布 FP16 60 TFLOPS；Apple 全系不公布 GPU TFLOPS；MI300A dense FP8 1.96 PFLOPs 对 5.3 TB/s。
- **硬件分区**：只有 Thor（MIG，按 TPC 切分，10 TPC / T4000 6 TPC）与 MI300A（SPX / TPX / CPX，按 XCD 切分）有官方公布的 GPU 计算分区；两者分区都不隔离内存（MI300A 文档原文 "no memory locality enforcement across partitions"；Thor MIG profile 为 `0gb` †）。Orin 规格表无 MIG；DGX Spark 官方文档未提 MIG；Apple、Strix Halo、Qualcomm、Intel 未公开任何 GPU 分区机制。
- **功耗档位**：Thor T5000 40–130 W、T4000 40–70 W；DGX Spark GB10 TDP 140 W；Strix Halo cTDP 45–120 W；AGX Orin 15–60 W；Lunar Lake 17–37 W；MI300A 550 W（峰值 760 W）。
- **CPU**：Thor 14 核 Neoverse-V3AE；DGX Spark 20 核（10 X925 + 10 A725）；Strix Halo 16 核 Zen 5（x86）；MI300A 24 核 Zen 4（x86）；AGX Orin 12 核 A78AE。

## 检索记录

- [S1] https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/
- [S2] https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai/ （2025-08-25；含 T5000/T4000 dense/sparse 表）
- [S3] † https://forums.developer.nvidia.com/t/agx-thor-jetpack-7-2-mig-reported-sm-counts-disagree-with-lgip-no-8-sm-profile-and-two-plain-instances-cannot-coexist/384830 （2026-10；MIG profile 与 SM 数，NVIDIA staff 确认 nvidia-smi SM 显示有 bug）
- [S4] † https://forums.developer.nvidia.com/t/clarification-on-cuda-core-and-tensor-core-counts-for-jetson-agx-thor/356355 （2026-01；NVIDIA staff：2560 CUDA cores，今后不公布 Tensor Core 数；20 SM 为楼主实测）
- [S5] https://forums.developer.nvidia.com/t/nvidia-jetson-t4000-and-nvidia-jetpack-7-1-now-available-accelerate-ai-inference-for-edge-and-robotics/356852 （2026-01-07）
- [S6] https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/
- [S7] https://nvidianews.nvidia.com/news/nvidia-announces-availability-of-jetson-agx-orin-developer-kit-to-advance-robotics-and-edge-ai
- [S8] † https://www.therobotreport.com/jetson-orin-nano-super-developer-kit-available/ （Orin Nano Super 2024-12-17）
- [S9] https://www.nvidia.com/en-us/products/workstations/dgx-spark/
- [S10] https://docs.nvidia.com/dgx/dgx-spark/hardware.html （6144 CUDA cores、TDP 140 W）
- [S11] † https://www.maginative.com/nvidia-starts-selling-dgx-spark-on-oct-15-hand-delivers-first-unit-to-elon-musk ；https://nvidianews.nvidia.com/news/nvidia-dgx-spark-arrives-for-worlds-ai-developers
- [S12] https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/
- [S13] https://support.apple.com/en-us/121555 （Mac mini 2024 技术规格）
- [S14] https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/
- [S15] https://support.apple.com/en-us/122211 （Mac Studio 2025 技术规格）
- [S16] https://www.apple.com/newsroom/2025/03/apple-reveals-m3-ultra-taking-apple-silicon-to-a-new-extreme/
- [S17] https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/
- [S18] https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/
- [S19] https://www.apple.com/macbook-pro/specs/ ；https://www.apple.com/mac-studio/specs/ ；https://www.apple.com/mac-mini/specs/
- [S20] https://www.apple.com/newsroom/2026/08/apple-introduces-new-mac-studio-with-m5-max-and-m5-ultra/
- [S21] https://www.apple.com/newsroom/2026/08/apple-unveils-a-more-powerful-mac-mini-featuring-the-all-new-m6-and-m5-pro/
- [S22] https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html
- [S23] https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo/ryzen-ai-max-plus-395.html （256 GB/s、FP16 60 TFLOPS、TDP 120 W）
- [S24] https://ir.amd.com/news-events/press-releases/detail/1232/amd-announces-expanded-consumer-and-commercial-ai-pc-portfolio-at-ces （2025-01-06；"up to 96GB available for graphics"）
- [S25] https://rocm.blogs.amd.com/artificial-intelligence/ryzen-uma-llm/README.html （2026-05-25；UMA 说明）；† https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html （gfx1151）
- [S26] https://www.amd.com/en/products/accelerators/instinct/mi300/mi300a.html
- [S27] https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/gpu-partitioning/mi300a/overview.html
- [S28] https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Product-Brief-ES-Snapdragon-X-Elite.pdf （DCN 87-71417-1 Rev F）
- [S29] † https://www.notebookcheck.net/Snapdragon-X2-Elite-Extreme-X2E-96-100-Snapdragon-X2-Elite-X2E-88-100-and-Snapdragon-X2-Elite-X2E-80-100-officially-announced-as-Qualcomm-s-new-laptop-chips.1123248.0.html
- [S30] † https://www.windowscentral.com/hardware/qualcomm/snapdragon-x2-elite-vs-x-elite ；https://www.tweaktown.com/news/107882/qualcomms-new-snapdragon-x2-elite-elite-extreme-soc-up-to-5-0ghz-with-up-to-18-cores/index.html ；官方页 https://www.qualcomm.com/laptops/products/snapdragon-x2-elite 为 JS 渲染，未能读取
- [S31] https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Snapdragon-8-Elite-Platform-Product-Brief.pdf ；https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Snapdragon-8-Elite-SM8750-3-AB-Product-Brief.pdf （SM8750-3-AB 为 2+5 核变体）
- [S32] † https://www.theregister.com/2024/10/22/qualcomm_snapdragon_8_elite/ ；https://androidcentral.com/phones/qualcomm-announces-snapdragon-8-elite-chip
- [S33] https://www.intel.com/content/www/us/en/products/sku/240961/intel-core-ultra-9-processor-288v-12m-cache-up-to-5-10-ghz/specifications.html
- [S34] † https://www.anandtech.com/show/21425/intel-lunar-lake-architecture-deep-dive-lion-cove-xe2-and-npu4 （128-bit on-package 内存接口）
- [S35] https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/ ；datasheet https://dam-cdn.nvd.orangelogic.com/AssetLink/tkno03fa645gb7f33w60rf200d330420.pdf （MAR25 版）
- [S36] GH200 架构白皮书 V1.21 https://dam-cdn.nvd.orangelogic.com/AssetLink/h125m07nuew832vthbov0q6b2rs8ta0q.pdf
- [S37] https://investor.nvidia.com/news/press-release-details/2023/NVIDIA-Grace-Hopper-Superchips-Designed-for-Accelerated-Generative-AI-Enter-Full-Production/default.aspx
- 另查阅：https://developer.nvidia.com/embedded/jetson-modules （仅系列级数字）
