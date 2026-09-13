# 第2节写作与实现核对说明

## 一句话论点与读者

面向计算光学和显微成像研究者，解释如何在固定十帧输入下，以均值初始体为锚点、融合方差和逐帧特征，通过强度/结构分离及有界均值结构梯度进行自监督训练。本节回答方法如何工作及如何复现，性能收益由后续Results建立。

用户已明确期刊、章节、贡献方向和指定运行，本次直接续写，不重复进行框架确认。目标为OE，不套用Nature期刊的章节或字数限制。

## 小节与段落分工

- 2.1：定义输入和光场算子；定义有限帧统计；从完整协方差关系说明所用平方PSF近似及其边界。
- 2.2：给出已保存初始重建实际使用的乘法更新式；说明均值/方差双体编码、逐帧集合编码和三尺度门控融合。
- 2.3：定义均值锚点的标定、非负残差映射和强度/结构分离；区分中间尺度beta与最终尺度a。
- 2.4：明确10/90不重叠划分、各损失项、stop-gradient路径和逐样本的梯度预算。
- 2.5：记录指定400步运行的训练规模、超参数、模型选择与实际推理流程。

实验硬件、组织制备和评价指标由后续实验章节承担；缺失项在Word文末单列，不填造参数或实验结果。

## 术语与符号表

| 固定术语/符号 | 定义 | 使用边界 |
|---|---|---|
| light-field microscopy (LFM) | 光场显微 | 延续Introduction已展开的缩写。 |
| mean-anchored | 以均值初始重建为输出锚点 | 不表示最终输出与均值初始体距离受限。 |
| M_I | 输入帧均值经三步乘法反投影重建得到的体 | 对应数据中的Mean-RL3 / g_mean。 |
| T_I | 输入帧方差经平方核三步重建后开平方得到的体 | 对应Taylor-RL3-sqrt；不重复开平方。 |
| H / H_2 | 光场前向算子 / 逐元素平方PSF的前向算子 | H_2不是H的复合。 |
| B | 实现中使用的反投影算子 | 不未经证明将其称作严格数学伴随。 |
| I / C | 十帧输入集 / 九十帧约束集 | 同一实例内不重叠；不同实例的集合可以重叠。 |
| beta / a | 残差校正前的锚点尺度 / 校正后的全局强度 | 两个尺度不能合并为同一个参数。 |
| q | 和归一化的非负结构体 | 非零体的和为1，零输入由数值下界处理。 |
| rho | 像素平均的SmoothL1损失 | 阈值为1。 |
| c_t | 脱离梯度图的自适应均值结构权重 | 在q空间控制梯度范数，不是固定loss权重1。 |

## 关键事实与依据

| 内容 | 已核对依据 | 状态 |
|---|---|---|
| 十帧输入、九十帧训练约束 | 指定运行YAML、run_contract.json、matlab_multivolume_dataset.py | 已核对；没有重训。 |
| 三步初始重建的真实更新式 | pilot_reconstruct_volume.m、Solver/deconvRL.m及冻结P01/subsets/subset_01.mat中的solver_update字段 | 已核对源码与保存记录；式(5)与保存记录一致。 |
| 三分支、16/32/64及8/16/32通道、mean/std池化、深度坐标提升 | encoder3d.py、set_encoder.py、gated_fusion.py、decoder3d.py | 已核对；不是attention/Transformer。 |
| mean anchor及其非负映射 | configurable_anchor_lfm_net.py、variance_anchored_lfm_net.py | 已核对指定配置激活的分支。 |
| beta0、a0及stop-gradient | mean_anchor_experiment.py的analytic_beta0和forward | 已核对。 |
| 方差/均值/TV项及梯度比例 | v3_compare_experiment.py的loss；self_supervised_losses.py | 已核对；TV不除以体素间距。 |
| 400更新、50步ramp、200步降学习率、参数量与最佳步数 | 指定运行YAML、run_contract.json、final_acceptance.json | 已核对；best_step=400。 |
| 散斑相关、噪声、真实样本高分辨收益 | 当前方法的适用条件及待补实验 | 不在本节写成已证实结果。 |

## 与早期初稿相比的重要技术修正

旧稿将初始重建概称为“Richardson–Lucy iterations”。代码中的deconvRL实际使用X乘以反投影测量与反投影预测之比，初始化为反投影测量；它不是通常书写的先在探测器域取比值再反投影的RL更新。新版直接给出式(5)，正文使用“multiplicative backprojection updates”，数据名称Mean-RL3/Taylor-RL3-sqrt在记录中保留。没有改动算法。

当前代码包含后续增加的可选网络和数值分支。本稿仅描述冻结配置中开启的mean_rl3、legacy_anchor_positive和mean_std路径；不引入800步新实验、轴向质量再分配、attention或额外细节分支。已抽查冻结数据的solver_update，而非仅凭当前源码推断早期生成过程。

## 待补事项

1. 物理假设：用照明相关长度/NA、噪声强度、PSF失配实验确认式(4)适用范围；如最终改变物理模型，应同步改公式和损失。
2. 正式实验：菠菜根及鼠脑切片的实际制备、样本层级、标记与采集参数、独立参考成像；补齐必要的组织来源及伦理信息。
3. 减帧与空间细节：固定比较条件及判据，完成十帧网络与高帧传统重建的比较；轴向采样间隔不等于轴向分辨率。
4. 复现：准确的PSF来源/标定、FFT边界及数值处理、软件版本、GPU型号、权重和代码发布位置。发布尚未发生，不承诺已有公开仓库。
5. 评价独立性：当前已参与开发比较的测试对象不能称为锁定方法后的最终盲测；增加独立测试和多个训练种子。

## 文献及格式

文献[2]、[5]、[7]沿用2026-09-12 Introduction的编号和核对记录。独立文件仅列本节所引三篇；合并全文时保留完整统一文献表。

- Broxton et al.：https://doi.org/10.1364/OE.21.025418
- Idier et al.：https://doi.org/10.1109/TCI.2017.2771729
- Taylor et al.：https://doi.org/10.1364/OPTICA.5.000345 ；原作者说明：https://vaziri.rockefeller.edu/brain-wide-3d-light-field-imaging-of-neuronal-activity-with-speckle-enhanced-resolution/
- OE审稿标准（2025-10版，本次核对）：https://opg.optica.org/resources/author/Optics_Express_Research_Article_criteria_Oct2025.pdf

五小节、15个编号公式和约1500词正文是本次组织选择，不是OE强制配额。已有架构图说明辅助了段落顺序，但当前可见图稿关联后续运行，本次没有把它未经核对地作为冻结400步模型的正式图插入。

后续可按“小节号/公式号＋需要调整的表述”反馈，保留已认可内容做局部修订。
