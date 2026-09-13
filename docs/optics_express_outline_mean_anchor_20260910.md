# Optics Express 论文框架：均值锚定的散斑照明光场三维重建

整理日期：2026-09-10。文档用途：论文组织、证据梳理与实验规划；不是已经完成实验的投稿正文。

本稿以用户指定的 `v3_mean_anchor_e3_mean100_400_20260909_run01_newbest` 中 **mean_anchor_e3_mean100** 为当前方法与结果来源。后续真实实验包括**菠菜根和鼠脑切片**。文中“已核实”表示配置、源码或现有评估记录支持；“待补充”表示需要开展实验或由作者确认。

## 一、论文定位与题目

**建议主线：利用均值重建锚点、散斑统计特征及受限的均值结构梯度，在少帧散斑照明光场成像中改善三维重建的形状保真度、深度分布和重复子集稳定性，并用植物与动物组织验证适用性。**

现阶段证据最充分的是重建质量与稳定性。将“超分辨”“轴向分辨率突破”“实时”“广泛泛化”写入标题，需要额外的分辨率、计时和独立实验支持。

英文题目首选：

**Mean-anchored self-supervised volumetric reconstruction for speckle-illumination light-field microscopy**

对应中文：**用于散斑照明光场显微成像的均值锚定自监督三维重建**。

若后续完成多帧数实验且结果支持，可使用强调帧数的备选：

**Mean-anchored self-supervised reconstruction from few speckle measurements in light-field microscopy**

建议在引言末围绕以下三点组织贡献，最终措辞需结合文献比较与补充实验确定：

1. **均值锚定的多分支重建。** 以 Mean-RL3 为非负输出的基础，同时保留 Taylor 方差重建和原始散斑集合特征；检验更换锚点对形状、深度与稳定性的影响。
2. **亮度与结构分开处理的统计约束。** 从输入帧解析校准亮度，利用归一化方差和有梯度上限的均值项约束结构，明确两类统计量在优化中的作用。
3. **从仿真到组织的验证体系。** 仿真量化形状、连续性、横向与轴向表现；菠菜根检验植物组织边界及管状结构，鼠脑切片检验复杂组织中的局部结构与层间串扰。真实实验完成前，第三点只能作为验证计划。

“物理自监督 LFM”本身已有相关研究，例如 SeReNet。因此创新论证应聚焦本工作的散斑输入、锚定方式与均值/方差结构约束，不能仅以“没有三维 GT 监督”为创新结论。[SeReNet 原始论文](https://www.nature.com/articles/s41592-025-02698-z)

## 二、与 Optics Express 投稿要求的对应

以下区分官方要求和本稿的组织建议，核对日期为 2026-09-10。

| 项目 | 官方指南 / 本稿处理 |
|---|---|
| 稿件类型 | 按 Research Article 组织；光学成像机制和重建证据贯穿全文。 |
| 模板 | 使用 Optica Publishing Group 的 Universal Manuscript Template，支持 Word 或 LaTeX；官方说明视觉样式并非初投稿的强制要求。 |
| 摘要 | 按当前 Style Guide 的**约 100 个英文词**规划，一段交代问题、方法、核心结果和结论。这里不是套用 200–250 词的通用摘要格式。 |
| 正文顺序 | Introduction → Methods → Results → Discussion → Conclusion；方法不放在全文结尾。 |
| 作者信息 | 预留姓名、单位、地址、邮箱；当前指南要求仅指定一名通讯作者。具体署名由作者团队确认。 |
| 文后内容 | Funding；Acknowledgments（如有）；Disclosures；Data availability；References。Disclosures 和 Data availability 必须具备，后者紧接前者。 |
| Funding | 正式出版中的资助信息由 Prism 提交信息生成，基金名称和编号应与投稿系统一致。 |
| 图表和引文 | 正文按出现顺序引用图、表和参考文献，使用官方引用样式；详细推导与补充对照可放 Supplement 1。 |
| LaTeX 提交 | 官方要求一个主 `.tex` 文件，注意图片文件名大小写与引用一致；完整提交所需资源。 |
| 篇幅 | 本稿建议先规划为约 10–12 个模板页、6 张主图、2–3 张表，再随内容调整；这是写作安排，**不是期刊规定的页数或图数上限**。 |

格式与文后安排依据：[官方模板入口](https://opg.optica.org/content/author/portal/item/templates-default)、[官方 Style Guide](https://opg.optica.org/content/author/portal/item/style-optica-styleguide/)、[官方投稿清单](https://opg.optica.org/content/author/portal/item/submission-checklist)、[Data availability 政策](https://opg.optica.org/content/author/portal/item/review-general-policies/)。

**Novelty and Impact Statement 的处理：**旧版官方投稿清单与 OE 的 2021 年审稿标准涉及 OE 的创新与影响声明；当前在线通用清单所列必需期刊未包含 OE。两处信息不一致。本稿预留一份简短声明的写作要点，正式提交时以 OE 在 Prism 中实际显示的字段为准，不将旧材料的字数限制直接写成当前已确认要求。[旧版官方清单](https://opg.optica.org/resources/author/SubmissionChecklist.pdf)、[OE 审稿标准](https://opg.optica.org/resources/author/Optics_Express_Research_Article_criteria_Sept_2021.pdf)、[当前在线清单](https://opg.optica.org/content/author/portal/item/submission-checklist)

**AI 辅助写作范围：**当前 Optica 政策允许组织建议和对作者原文的编辑，但对生成投稿正文有限制，并要求如实记录相关图像、代码生成用途。本文件用于框架规划；正式科学论述应由作者依据实验和文献撰写、核验，不应将框架中的待验证论点当作已完成结果。[官方 AI 使用政策](https://opg.optica.org/content/author/portal/item/review-general-policies/)

## 三、当前方法和结果的冻结口径

### 3.1 本稿采用的方法版本

| 项目 | 已核实内容 |
|---|---|
| 实验目录 | `/workspace/xyx/speckle_illumination_net/outputs/v3_mean_anchor_e3_mean100_400_20260909_run01_newbest` |
| 方法标识 | `mean_anchor_e3_mean100` |
| 主结果 | 第 400 次优化更新后的 final，即 `mean_anchor_e3_mean100/checkpoint_last.pt` |
| best 口径 | 完整实验报告记载本轮 best 也位于第 400 步；本文主表仍统一标为 final，不混入其它目录的历史 best。 |
| 训练模式 | 多对象共享网络训练；不沿用早期 README 中“对每个体独立优化”的方法描述。 |
| 输入 | 10 帧原始散斑图，以及从同一组 10 帧得到的 Mean-RL3、Taylor-RL3-sqrt。 |
| 训练约束 | 另外 90 帧的均值与方差；三维 GT 不进入当前训练损失与 checkpoint 选择分数。 |
| 数据划分 | P01–P11 训练，V01–V03 验证，T02–T04 测试；每对象 10 个固定子集，共 110/30/30 个对象–子集实例。 |
| 网络 | Mean 与 Taylor 两路 3D 编码器；共享 2D CNN + mean/std 集合聚合 + 深度提升；多尺度门控融合；3D 残差解码器。 |
| Set 分支 | `set_encoder_type: mean_std`，没有启用 Set Transformer。E3 是实验目标的命名，不能据此写成 Transformer 模型。 |
| 输出锚点 | `reconstruction_anchor: mean_rl3`；Taylor 分支仍使用已保存的开方表示，不能重复开方。 |
| 结构约束 | 归一化 log 方差损失 + 受限的归一化 mean 结构梯度 + 弱 TV。普通 mean 项只更新亮度增益。 |
| “100% mean”含义 | 每个样本在归一化重建 q 处，mean 结构梯度的范数上限相对于方差梯度为 1；并非固定 loss 权重 1，也不是实际占比必为 100%。 |
| 训练预算 | 随机种子 20260901，400 次 optimizer updates，global batch=8；前 50 步渐增 mean 结构梯度预算，第 200 步后学习率下降。400 步不是 400 epochs。 |
| 数值实现 | FP32，关闭 AMP 和 TF32；配置记录为 Adam、六张 GPU，参数量 1,188,377。 |
| 仿真网格 | 260×260×10，XY 采样约 1.12245 µm；z=10–100 µm，间隔 10 µm。采样间隔不等于光学分辨率。 |
| 光学参数状态 | 本轮配置为 detection NA=0.15、illumination NA=0.05；真实系统 NA 仍待确认，不从文件名推定所有实机参数。 |

本稿优先级为：**用户指定运行 > 该运行配置/记录/源码快照 > 项目通用基线说明**。现有 `CURRENT_BASELINE.md` 仍描述前一代 Taylor 锚定基线，不用于替代本稿方法定义。

### 3.2 可直接用于框架和后续主表的仿真结果

下表来自指定目录 `analysis/quality_summary.csv` 的 `object_macro` 行。先在每个测试对象内部汇总 10 个子集，再对 T02、T03、T04 等权平均；**不是 30 个独立样本**。

| 方法 | 强度尺度对齐 NRMSE↓ | GT 轴向分布 W1 / µm↓ | XY 支持外质量占比↓ | XY-MIP SSIM↑ |
|---|---:|---:|---:|---:|
| Mean-RL3 | 0.9300 | 21.95 | 16.991% | 0.74595 |
| Taylor-RL3-sqrt | 0.9457 | 24.53 | 20.178% | 0.72437 |
| Taylor-anchor network | 0.8169 | 5.22 | 0.378% | 0.74746 |
| **Mean-anchor network** | **0.7828** | **4.64** | **0.249%** | 0.74714 |

“尺度对齐”是在评价时求一个非负全局强度系数，不是空间配准，也不是推理时用 GT 校准亮度。XY 支持外质量是根据 GT 的 XY 投影阈值与膨胀掩膜计算，不能改称完整三维支持外能量或实验背景率。

| 子集稳定性指标 | Taylor anchor | Mean anchor |
|---|---:|---:|
| 归一化形状相对子集离散度↓ | 0.2551 | 0.2412 |
| 子集之间轴向分布 W1 / µm↓ | 0.364 | 0.251 |

当前支持的论点：在相同 10 帧 RL3 输入方式、同一初始化种子和 400 步预算下，更换为 mean anchor 改善了测试对象宏平均的形状误差、轴向分布误差、XY 支持外质量及子集稳定性。SSIM 没有同步提升，不能概括为“所有指标均优”。

### 3.3 必须保留的结果边界

| 现有发现 | 论文中的解释与安排 |
|---|---|
| T02 中心线恢复率 0.999→0.971，假断率 0.031→0.169 | 细管连续性存在代价，进入局部结果和 Discussion；菠菜根与鼠脑线状结构实验必须检查断裂。 |
| T03 三线分开率两组均为 0.667 | 尚不支持“相对 Taylor anchor 提高了横向分辨率”的结论。 |
| T03 局部深度 W1 3.98→4.28 µm | 宏平均改善不代表每类局部结构都改善。 |
| T04 的 20–40 µm 间距分组正式双层分开率为 0 | 不宣称该范围轴向双层可分辨；轴向分布 W1 改善与分辨率改善须分别论证。 |
| 相邻 10 µm 深度层 | 报告原生层能量与定位表现，不以插值曲线的双峰证明分辨率。 |
| 有效修正 L2 范数平均为校准 Mean 基础体的 19.03 倍 | 最终形状仍高度依赖网络修正；“anchor”不能写成“小残差、几乎保留原图”的保证。 |
| 当前主要比较只有一个训练随机种子 | 结果定位为当前设置的初步证据；补充多种子和独立对象后再作稳健性结论。 |

以上局部数值来自指定运行的 `FINAL_REPORT_ZH.md`，不是新增计算或新增实验。

## 四、建议采用的正式论文目录

```text
Title
Authors and affiliations
Abstract

1. Introduction

2. Methods
   2.1. Image formation and speckle statistics
   2.2. Mean-anchored reconstruction network
   2.3. Self-supervised statistical constraints and gradient balancing
   2.4. Simulation data, training, and inference
   2.5. Optical setup and biological sample preparation
   2.6. Baselines and evaluation protocol

3. Results
   3.1. Reconstruction accuracy on simulated three-dimensional objects
   3.2. Lateral structure, axial localization, and subset stability
   3.3. Ablation of the anchor and statistical constraints
   3.4. Experimental reconstruction of spinach roots
   3.5. Experimental reconstruction of mouse brain sections
   3.6. Frame-budget and computational performance [if completed]

4. Discussion

5. Conclusion

Funding
Acknowledgments [if applicable]
Disclosures
Data availability
References

Supplement 1 [separate file]
Visualizations [if provided]
```

### Abstract：约 100 个英文词的内容槽位

建议按 4–5 句组织，不在摘要中逐项介绍网络层数：

1. **问题（约 15–20 词）：**少帧散斑测量下，光场三维反演的形状和深度稳定性。
2. **方法（约 30–35 词）：**mean anchor、多分支特征及均值/方差统计自监督。
3. **量化结果（约 20–25 词）：**写一项最能代表贡献的独立仿真结果，说明比较基准；待扩大验证后锁定数值。
4. **真实实验（约 15–20 词）：**菠菜根与鼠脑切片的实际观察，全部留待实验完成后填入。
5. **结论：**限定到已经验证的成像条件与用途。

当前不生成“已在鼠脑切片实现某种性能”的完成式摘要句。标题与摘要中也不加入尚未证实的超分辨倍数或速度倍数。

### 1. Introduction

建议四段，总体约 700–900 英文词，篇幅仅为写作建议。

**第一段：成像问题。**介绍光场成像的三维信息编码，以及在有限测量和复杂组织中保持结构与深度可信度的需求。菠菜根和鼠脑切片作为后文验证对象自然引出，不预设生物学发现。

**第二段：相关方法。**按三条线梳理：光场波动光学模型与迭代重建；随机/散斑照明的统计重建；监督和物理自监督的学习式 LFM。明确不同方法的光学测量和监督条件，避免把任意二维图像增强网络当作等价 LFM 对照。

**第三段：本研究提出的问题。**以代码所采用的近似为背景，提出可检验假设：短序列统计估计会影响重建稳定性；均值与方差不宜不加区分地承担亮度、结构两个任务；更换输出锚点可能改变优化和跨子集表现。假设的普遍性需要文献和本稿实验证据，不能从当前一个运行反推为一般定律。

**第四段：本文方案与贡献。**概括 mean anchor、亮度与结构处理、受限 mean 结构梯度、对象级划分及组织验证。避免声称“首次物理自监督光场重建”。

### 2. Methods

#### 2.1. Image formation and speckle statistics

写作目标：让读者知道网络面对的是何种光学反问题，以及统计近似保留了什么。

建议先定义概念模型：

\[
y_n=\mathcal H(f\odot s_n)+\eta_n,\qquad n=1,\ldots,N,
\]

其中 \(f\) 为待估计的非负三维荧光分布，\(s_n\) 为第 n 次照明，\(\mathcal H\) 为光场前向算子，\(\eta_n\) 表示测量噪声。实际稿件需写明线性响应、平均照明标定及物体在采集期间稳定等假设。

介绍当前 \(\mathcal H\) 的周期位置相关 PSF；49×49 表示一个微透镜周期内的相位位置，不是 2401 个微透镜。写清深度采样、PSF 轴序、原点及强度标定；算子实现验证可放补充材料。

定义同一帧集合内的均值和样本方差：

\[
\mu_N=\frac1N\sum_n y_n,\qquad
V_N=\frac1{N-1}\sum_n(y_n-\mu_N)^2.
\]

当前方法使用的结构统计近似可写为：

\[
\widehat\mu=\mathcal H(f),\qquad
\widehat V\approx\mathcal H_2(f^2),
\]

其中 \(\mathcal H_2\) 由逐元素平方的 PSF 核组成，不是 \(\mathcal H\circ\mathcal H\)。上述表达采用当前配置的标度与零附加噪声项；正式实机模型应说明必要的照明系数、暗场、散粒和读出噪声处理。该 Taylor 近似未包含完整的有限宽度散斑协方差交叉项，不应表述为精确完整协方差模型。

说明 Mean-RL3 与 Taylor-RL3-sqrt 如何分别从均值与方差生成；RL3 表示三次迭代，后者为保存的开方幅度体。

#### 2.2. Mean-anchored reconstruction network

按数据流说明，建议对应 Fig. 1：

1. Mean-RL3 与 Taylor-RL3-sqrt 分别进入 3D 编码器；通道为 16/32/64，XY 下采样，保持原生 z 层数。
2. 10 张去均值原始帧进入共享 2D CNN；对帧维的非线性特征做 mean/std 聚合，再结合实际 z 坐标提升为 3D 特征。
3. 通过三尺度门控融合输入 3D 解码器，得到残差 \(R_\theta\)。集合聚合对帧排列不敏感，不引入帧序号编码。
4. 以 \(A=\beta M_{10}\) 为锚点形成非负中间输出 \(f_0=\Phi(A,R_\theta)\)：

\[
\Phi(A,R)=
\begin{cases}
A+R,&R\ge0,\\
A\exp\!\bigl(R/\max(A,\epsilon)\bigr),&R<0.
\end{cases}
\]

末层零初始化使初始输出等于锚点；不限制训练后的正残差必须很小。解释特征归一化与物理强度标定的区别。

若展示 Gate 图，只解释其控制特征注入的位置和幅度；模块是否改善重建应通过消融判断。

#### 2.3. Self-supervised statistical constraints and gradient balancing

这是最值得展开的方法节，建议分三个连续段落。

**亮度与结构表示。**对非负中间输出进行单位总量归一化：

\[
q=\frac{f_0}{\sum f_0},\qquad \hat f=a q.
\]

实际实现对分母设数值下限。输入 10 帧的均值提供解析初始增益：

\[
a_0=\max\!\left(0,\frac{\langle\mathcal H(q),\mu_{10}\rangle}
{\|\mathcal H(q)\|_2^2}\right),\qquad
a=a_0[1+0.2\tanh\gamma].
\]

计算 \(a_0\) 和普通 mean 项预测时对 q 停止梯度；因此普通 mean 亮度损失只更新共享标量 \(\gamma\)。初始锚点缩放 \(\beta\) 与最终输出增益 \(a\) 是两个不同变量，附录可给完整算法步骤。

**统计损失。**以另外 90 帧得到的 \(\mu_{90},V_{90}\) 作为训练约束；令 \(\mathcal N(u)=u/\operatorname{mean}_{xy}(u)\)，令 \(\rho\) 为取平均的 SmoothL1：

\[
L_V=\rho\!\left(\log[\mathcal N(\mathcal H_2(q^2))+\epsilon],
\log[\mathcal N(V_{90})+\epsilon]\right),
\]

\[
L_{M,\mathrm{shape}}=\rho\!\left(\mathcal N(\mathcal H(q)),
\mathcal N(\mu_{90})\right).
\]

普通亮度损失为 \(L_{M,\mathrm{gain}}=\rho(a\mathcal H(\mathrm{sg}(q))/s,\mu_{90}/s)\)，其中 \(s=\operatorname{mean}|\mu_{90}|\) 且作数值下限处理，\(\mathrm{sg}\) 表示停止梯度。保留其与 shape mean 的不同梯度路径。

**受限 mean 结构梯度。**按当前实现，逐样本在 q 处计算：

\[
c_t=\mathrm{sg}\!\left[
\min\!\left(1,
b_t\frac{\|\nabla_q L_V\|_2}
{\|\nabla_q L_{M,\mathrm{shape}}\|_2+\epsilon_g}\right)
\right],\qquad b_t=\min(t/50,1),
\]

其中 t 从第一次 optimizer update 开始计数。总训练目标为：

\[
L=L_{M,\mathrm{gain}}+L_V+c_tL_{M,\mathrm{shape}}
+10^{-5}\,\mathrm{TV}_{z:0.5}\bigl(q\,\mathrm{sg}(a)\bigr).
\]

明确这是 q 空间的梯度范数约束，不是所有网络参数梯度的逐项上限。可在 Supplement 1 中给预算实际轨迹、梯度夹角及 ramp 配置。

checkpoint 选择采用验证对象等权的 mean+variance+TV 分数，不包含训练期间的受限 shape mean 项，不使用 GT 指标。当前 YAML 的 `lambda_mean: 0` 是兼容字段，不能据此将该实验描述为完全没有均值损失；必须连同实验适配器复现。

#### 2.4. Simulation data, training, and inference

写清场景生成、散斑生成、100 帧如何形成每组 10/90 划分、两种 RL3 输入的来源与尺度；仿真生成器和训练物理近似是否一致，需要在正式复现实验中说明。

给出 11/3/3 的对象级划分及每对象 10 个子集。来自同一对象的所有帧、子集和派生重建保留在同一数据集分区；不同子集之间允许重用帧，因此不能把它们当成独立生物重复。

描述推理流程：新对象仅用 10 帧计算输入统计量与 RL3 初步重建，经过固定网络和输入亮度标定得到体数据。另 90 帧只可在训练时作为约束，或在独立评估阶段作统计检查，不能进入所宣称的 10 帧推理。

列出网络/标量学习率、下降时点、batch、随机种子、400 次更新及选择规则。若增加实机微调，应单列直接迁移与微调两条协议，并说明微调消耗的测量帧数。

#### 2.5. Optical setup and biological sample preparation

此节先搭结构，参数留空等待实验记录，**不从仿真文件名填实机数据**。

光路需填写：激发/发射波段，散斑产生与切换方式，物镜倍率与 NA，照明 NA，微透镜参数，相机及像元，放大倍率，曝光和每帧时间，样品面功率或剂量口径，PSF 标定流程，重建范围及噪声校准。

菠菜根需填写：样品来源与状态、固定/切片或整根制备、标记或自发荧光来源、封片介质、厚度、独立根数及每根视野数；根据实际可见结构定义细胞壁、组织边界或管状区域，不能仅凭形状给出未经验证的组织学命名。

鼠脑切片需填写：动物与组织来源、脑区、固定方式、切片厚度、标记通道、封片介质、独立动物数、每只动物切片数及视野数。根据实际样本来源补充适用的伦理审批机构和编号，或准确说明商业/共享存档组织来源，不填虚构审批号。

两类样本都应说明是否采集配准的独立三维参考，例如共聚焦 z-stack。参考方法若分辨率或配准精度有限，应给出其边界；高帧数计算重建统一称为 reference，不能自动认定为 ground truth。

#### 2.6. Baselines and evaluation protocol

主要比较应包含同一 10 帧下的 Mean-RL、Taylor-RL-sqrt、Taylor-anchor network 和 Mean-anchor network。现有 RL3 是网络输入基线；投稿时还应增加通过验证数据选择合理迭代数的传统重建，避免仅凭“网络优于三步初步重建”证明方法先进性。

视可实现条件增加与本系统前向模型匹配的统计重建或物理自监督对照。若纳入 AlgoRIM，先确认其输入、模型和已完成输出与本系统兼容；不把已准备的输入栈写成已完成的 AlgoRIM 基线。

核心评价量：

- **仿真总体质量：**强度尺度对齐 NRMSE、未对齐误差（补充）、GT 轴向 W1、XY 支持外质量、SSIM；说明每个量的定义和标度。
- **局部结构：**线对/三线分开率、细管中心线恢复率、假断率与假连接率；各阈值和 ROI 在比较前固定。
- **轴向：**原生层剖面、轴向定位偏差、双层分开率；将定位误差、分布误差和两点分辨率分别报告。
- **实验图像：**标定样本的 PSF/FWHM 或预先定义的线对判据；组织结构连续性、局部对比度、与独立参考的一致性。组织轮廓变窄本身不能证明分辨率提升。
- **稳定性：**重复采集/不同帧子集下的形状离散度及轴向变化，区别测量重复和训练随机种子重复。

仿真统计按对象等权汇总；组织实验按独立根/独立动物报告层级，切片、视野、ROI 为嵌套测量。给对象散点和变异范围；独立样本较少时不以大量 ROI 或子集制造显著性。

图像比较需统一 ROI、物理坐标、比例尺及显示规则，保留原生 XY/XZ/YZ 视图。若另做形状归一化视图，应与强度视图区分，并说明是整幅体归一化；不逐层独立拉伸以隐藏深度伪影。

### 3. Results

#### 3.1. Reconstruction accuracy on simulated three-dimensional objects

**要回答的问题：**在相同输入帧数和训练预算下，mean anchor 改善了哪些三维质量指标？

以 T02 细管、T03 三线和 T04 轴向双层场景组织 Fig. 2，显示 GT、两种 RL3、Taylor anchor 和 Mean anchor 的投影、切面和固定局部放大图。Table 1 使用第 3.2 节已有数值，并补逐对象结果。

结论重点为宏平均形状、深度分布与支持外质量的变化；保留 T03 对象上部分指标退步和 SSIM 基本不变的事实。形状 NRMSE 仍为 0.7828，不应称为“近乎精确恢复”。

#### 3.2. Lateral structure, axial localization, and subset stability

**要回答的问题：**总体改善是否对应真实局部结构恢复，是否能在不同 10 帧子集中重复？

Fig. 3 把横向线剖面、细管连续性及轴向双层剖面并列；给失败例，尤其是细管假断和双层未分开。Fig. 4 的一部分显示逐对象子集散点与两种稳定性指标。

当前位置可报告三对象的子集稳定性改善，但不宣称同时提升横向、轴向分辨率。若补充微球或标定线对实验支持进一步结论，再更新本节标题和结果叙述。

#### 3.3. Ablation of the anchor and statistical constraints

**要回答的问题：**增益来自均值锚点、统计目标，还是某个特征分支？

建议按优先级补齐下表，训练预算、数据划分、初始化种子和评价保持一致。

| 比较 | 目的 | 当前状态 |
|---|---|---|
| Taylor anchor vs Mean anchor，均采用 E3+mean≤100% | 隔离锚点选择 | 指定目录已有主对照 |
| Mean anchor：mean 结构梯度关闭 vs 开启 | 判断受限 mean 结构约束的作用 | 需要在当前锚点下统一补齐 |
| Mean anchor：受限 mean vs 固定权重 mean | 检验“限制梯度”的必要性 | 待补 |
| Mean anchor：去 Taylor 分支 | 判断方差信息的额外贡献 | 待当前版本统一对照 |
| Mean anchor：去 Set 分支 / 去 Gate | 判断原始帧集合与门控贡献 | 待当前版本统一对照 |
| 当前完整模型的多训练种子 | 排除单次初始化偶然性 | 待补 |

历史不同目录或不同锚点的消融仅用于选方案，不直接拼接为当前方法的公平消融。Fig. 4 和 Table 2 以重建指标为核心，梯度轨迹为机制解释。

#### 3.4. Experimental reconstruction of spinach roots

**验证角色：**在真实植物样本中检验组织边界、管状/连续结构、深度串扰及短序列重复性。

Fig. 5 预留：样本和视野定位图；10 帧均值或代表帧；Mean-RL、Taylor-RL、两种网络及独立参考；同一 ROI 的 XY/XZ/YZ、局部放大和预设线剖面；按独立根汇总的指标。

优先回答：细长结构是否被截断；邻近边界是否发生假连接；不同子集能否得到相同深度和形状；网络补出的结构能否在独立参考中找到依据。

**已有预实验的定位：**指定运行有 `spinach_root_transfer/REPORT_ZH.md`，只支持探索性迁移比较。Mean anchor 对 90 帧均值/方差的 NRMSE 为 0.3721/0.6775，优于 Taylor anchor 的 0.4001/0.7320；但 Mean-RL3 为 0.3399/0.6700，前向统计残差并非网络最好。数据清单还记录了逐帧最大值归一化与 uint8 量化问题。上述数值不能证明真实三维深度正确，先作为后续采集和校准的依据，不替代正式组织验证。

正式采集优先保存相机原始线性强度、固定标定关系和暗场；保留逐帧光强变化供统计建模。逐帧归一化会改变需要利用的散斑波动，处理方案必须在实验方法中透明说明。

#### 3.5. Experimental reconstruction of mouse brain sections

**验证角色：**检验植物样本之外的组织适用性，以及复杂三维分布中的假结构和层间串扰。

Fig. 6 预留：脑区与视野定位、标记通道说明、各方法三维重建、独立参考、固定 ROI 的截面/剖面、按独立动物组织的量化结果。

目标结构以实际标记为准：核染优先评价相邻核分离、质心定位和深度归属；神经元形态标记可评价胞体及可验证的粗突起连续性；血管标记可评价管径、连续性和交叉处层间归属。不要在目前尺度下预设能够识别树突棘、突触或超微结构。

优先使用在实验评估前固定的模型。如果必须适配新 PSF 或做实机微调，报告适配过程及所用样本，并保留独立动物/切片作为测试；不要把观察测试效果后的调参写成零样本泛化。

切片实验的结论限于对应离体组织条件，不外推到活体功能成像或神经活动测量。

#### 3.6. Frame-budget and computational performance（补齐后保留）

若论文要突出 few-frame，建议在 N=5/10/20/50 等预算下作系统评价；这是建议实验点，不是已有结果。每种 N 都从对应输入子集重新计算 Mean/Taylor 输入和统计量，并明确固定网络测试还是重新训练。

区分固定每帧曝光与固定总曝光量。更少帧不自动等于按同一倍数降低剂量或提高体积帧率。测试对象的参考帧不能被提前用于输入、适配或参数选择。

计时至少分成：采集、统计量计算、两种 RL 初步重建、网络前向、全幅拼接/输出。报告完整重建时间和硬件，不能仅将 CNN 前向时间当作端到端速度。若未完成此组实验，将本节移入补充材料或删除，主文保留“当前使用 10 帧”的事实。

### 4. Discussion

建议四个段落：

1. **机制解释。**均值锚点改变输入基础和优化起点，亮度/结构分开处理及梯度上限有明确算法定义；实际改善机制仍需消融支持，不把当前性能差异写成可辨识性定理。
2. **光学意义。**解释当前贡献如何改善散斑光场重建的形状和深度可靠性，以及和既有 LFM/RIM、自监督方法的关系；与单纯图像清晰化区分开。
3. **局限。**讨论 Taylor 统计近似、PSF/NA 失配、噪声和光强漂移、残差主导、细管断裂、未通过的双层分辨、单种子和独立样本量有限。
4. **应用范围。**根据菠菜根和鼠脑切片的实际结果定义适用条件，说明从离体静态组织到动态样品还需要哪些采集稳定性与速度验证。

### 5. Conclusion

建议一个短段，按“方法—经验证的主要结果—适用范围”收束。只概括正文已有证据，不新增数字，不将采样间隔写成分辨率，不用“所有指标最优”掩盖局部代价。两种组织实验完成前，结论中保留相应占位。

## 五、主图、表格与补充材料安排

| 编号 | 图题建议 | 主要内容 | 状态 |
|---|---|---|---|
| Fig. 1 | Optical model and mean-anchored reconstruction framework | 光路、10/90 训练划分、三分支网络、q/a 与统计约束；区分训练和推理箭头 | 方法可画，实机参数待填 |
| Fig. 2 | Quantitative reconstruction of simulated 3D objects | T02/T03/T04 对照、固定切面/局部图、总体指标 | 已有源数据 |
| Fig. 3 | Local structure and axial reconstruction assessment | 横向线、细管连续性、原生轴向双层、失败例 | 已有局部评估，标定实验待补 |
| Fig. 4 | Ablation and reconstruction stability | 锚点、结构梯度、分支消融；对象级子集散点 | 锚点/稳定性已有，其它待补 |
| Fig. 5 | Experimental reconstruction of spinach roots | 全视野、截面、局部轮廓、独立参考及重复统计 | 已有探索性结果，正式验证待补 |
| Fig. 6 | Experimental reconstruction of mouse brain sections | 组织视野、三维截面、目标结构与独立参考、动物级量化 | 待实验 |
| 可选 Fig. 7 | Reconstruction performance versus measurement budget | 帧数、曝光条件、质量和端到端耗时 | 完成后决定主文或补充 |

| 表格 | 内容 |
|---|---|
| Table 1 | 仿真主要对照：每对象及宏平均质量，清楚标明训练种子和子集数 |
| Table 2 | 消融及稳定性：统一数据和训练预算 |
| Table 3（可选） | 真实样本、帧数、重复层级、实验指标、端到端运行时间；内容过多则拆到补充 |

Supplement 1 建议包含：PSF/算子校验；所有网络层和超参数；数据生成与对象清单；全部子集指标；梯度限制伪代码与轨迹；多随机种子；补充局部结构和失败案例；曝光/噪声/失配试验；实机校准、配准与完整 ROI；端到端计时细项。

如制作逐层浏览或三维旋转视频，使用真实重建数据并标注 z、比例尺及方法，按官方要求编号为 Visualization。主文应独立支撑主要结论，不能把决定性失败或关键方法完全移到补充材料。[OE 审稿标准](https://opg.optica.org/resources/author/Optics_Express_Research_Article_criteria_Sept_2021.pdf)

## 六、补实验优先级与投稿材料

### 优先补齐的证据

1. **系统与数据标定。**确认实机波段、检测/照明 NA、PSF、样品面像素和原始强度预处理；解决菠菜根已有预实验中的归一化/量化解释问题。
2. **真正独立的仿真测试与多种子。**当前 T02–T04 的结果已经参与方法比较；最终定型后建议再增加未用于选方案的对象或新的独立测试集，并明确当前结果属于开发阶段比较。
3. **公平传统对照与核心消融。**补合理迭代预算的 RL，以及当前 mean anchor 下的 mean 结构项开关/梯度限制对照；其余分支消融按贡献重点安排。
4. **两类组织的正式实验。**优先保证原始数据、独立参考和跨样本重复；独立根/动物数量根据可行性及预实验变异设计，不把 ROI 数代替样本量。
5. **横向与轴向标定。**若希望声称分辨率提升，使用匹配的标定样本、预定判据和原生深度采样验证。
6. **帧数与速度。**若 few-frame/快速重建是标题或摘要卖点，补齐端到端预算实验证据。

### 文后和投稿系统的占位内容

- **Funding：**资助机构与项目编号，待作者提供并核对 Prism。
- **Acknowledgments：**实际实验、样本、设施及技术支持；如有适用的 AI 辅助记录，如实填写其具体用途。
- **Disclosures：**作者确认有无利益冲突后填写，不预设所有作者均无冲突。
- **Data availability：**说明原始帧、PSF、数据划分、预处理、配置、模型和评价代码的获取方式；若存入公开仓库，填写可用链接/DOI并引用；若不公开，按真实情况说明申请方式或限制。不能使用“未生成或分析数据”的声明。
- **Cover letter：**简要说明光学问题、主要贡献、已完成验证及期刊适配性，作为建议准备项。
- **Novelty and Impact Statement：**预留“现有方法缺口—本工作的具体改变—新增证据—对光学成像的意义”四个要点；是否必填及字数以提交时字段为准。

## 七、引言优先核对的文献入口

以下是起始阅读清单，不是已完成的系统文献综述；正式参考文献应从出版页面导出准确作者、卷页和 DOI，逐条核验正文所支持的论点。

| 文献 | 本文中的用途 |
|---|---|
| Broxton et al., “Wave optics theory and 3-D deconvolution for the light field microscope,” Optics Express 21, 25418–25439 (2013) | 光场波动光学与三维反卷积基础；[原论文全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC3867103/) |
| Mangeat et al., “Super-resolved live-cell imaging using random illumination microscopy,” Cell Reports Methods 1, 100009 (2021) | 随机照明与统计重建背景；[出版页面](https://www.sciencedirect.com/science/article/pii/S2667237521000096) |
| Lu et al., “Virtual-scanning light-field microscopy for robust snapshot high-resolution volumetric imaging,” Nature Methods 20, 735–746 (2023) | 光场高质量体重建的相关路线；[原论文](https://www.nature.com/articles/s41592-023-01839-6) |
| Lu et al., “Physics-driven self-supervised learning for fast high-resolution robust 3D reconstruction of light-field microscopy,” Nature Methods 22, 1545–1555 (2025) | 物理自监督 LFM 的重要相关工作与贡献边界；[原论文](https://www.nature.com/articles/s41592-025-02698-z) |

此外按正式稿重点补：实际使用的 RL/方差重建来源、集合聚合依据、相应分辨率评价方法，以及与实机配置兼容的近期重建方法。只有实际运行过的算法才能列为定量对照。

## 八、本稿的本地证据索引

下列链接均指向本次指定运行或实际核对的源码。部分运行清单内部仍写着不带 `_newbest` 的原始路径，这是迁移保留的来源字段；本次读取的是用户指定 `_newbest` 目录中的文件，后续复现打包时应统一修正路径映射。

- [运行配置](/workspace/xyx/speckle_illumination_net/outputs/v3_mean_anchor_e3_mean100_400_20260909_run01_newbest/mean_anchor_e3_mean100/config_used.yaml)
- [运行契约与参数量](/workspace/xyx/speckle_illumination_net/outputs/v3_mean_anchor_e3_mean100_400_20260909_run01_newbest/mean_anchor_e3_mean100/run_contract.json)
- [正式仿真划分](/workspace/xyx/speckle_illumination_net/outputs/v3_mean_anchor_e3_mean100_400_20260909_run01_newbest/dataset_v3_frozen_view/dataset_splits.json)
- [总体对照报告](/workspace/xyx/speckle_illumination_net/outputs/v3_mean_anchor_e3_mean100_400_20260909_run01_newbest/REPORT_ZH.md)
- [完整局部结果与解释](/workspace/xyx/speckle_illumination_net/outputs/v3_mean_anchor_e3_mean100_400_20260909_run01_newbest/FINAL_REPORT_ZH.md)
- [质量汇总原表](/workspace/xyx/speckle_illumination_net/outputs/v3_mean_anchor_e3_mean100_400_20260909_run01_newbest/analysis/quality_summary.csv)
- [子集稳定性原表](/workspace/xyx/speckle_illumination_net/outputs/v3_mean_anchor_e3_mean100_400_20260909_run01_newbest/analysis/stability_per_object.csv)
- [验证曲线原表](/workspace/xyx/speckle_illumination_net/outputs/v3_mean_anchor_e3_mean100_400_20260909_run01_newbest/analysis/supplemental/validation_curve.csv)
- [Mean-anchor 适配器快照](/workspace/xyx/speckle_illumination_net/outputs/v3_mean_anchor_e3_mean100_400_20260909_run01_newbest/source_snapshot/tools/mean_anchor_experiment.py)
- [损失与梯度限制快照](/workspace/xyx/speckle_illumination_net/outputs/v3_mean_anchor_e3_mean100_400_20260909_run01_newbest/source_snapshot/tools/v3_compare_experiment.py)
- [锚点选择实现快照](/workspace/xyx/speckle_illumination_net/outputs/v3_mean_anchor_e3_mean100_400_20260909_run01_newbest/source_snapshot/models/configurable_anchor_lfm_net.py)
- [菠菜根探索性迁移报告](/workspace/xyx/speckle_illumination_net/outputs/v3_mean_anchor_e3_mean100_400_20260909_run01_newbest/spinach_root_transfer/REPORT_ZH.md)
- [菠菜根数据及预处理清单](/workspace/xyx/speckle_illumination_net/outputs/v3_mean_anchor_e3_mean100_400_20260909_run01_newbest/spinach_root_transfer/manifest.json)

主模型 `checkpoint_last.pt` 文件 SHA256：`e1b7be665b108c4ddbe0e6e98b0197db8a3b321b8089a0091fa52ab5c8e7764e`。本次仅核对文件与已有记录、撰写框架，没有重新训练网络或重算重建结果。
