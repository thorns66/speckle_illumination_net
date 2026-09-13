# 第3节写作与证据说明

## 一句话论点及边界

以相同输入、比较条件和明确空间细节判据，评价十帧重建的结构保真度；再通过质量匹配的高帧对照、真实样本参考成像和独立重复建立减帧与高分辨结论。本节是实验设置和评价方法，不提前写Results。

用户已确认投稿OE、英文标题、贡献重点和指定mean_anchor运行，并授权在稿中标注待补实验，因此直接续写。Word中的将来时与[TO SUPPLY]表示未完成的实验或缺少记录，不代表已采集数据。

## 小节结构

1. 3.1 Numerical data and simulation conditions：数据划分、三个评价对象、体散斑生成、采样和当前无探测器噪声设置。
2. 3.2 Optical setup and biological specimens：真实系统、采集记录、菠菜根与鼠脑切片、独立参考成像和明确的待补参数。
3. 3.3 Comparison methods and measurement budgets：四种现有对照、合理迭代预算、少帧/高帧比较、质量目标、消融与计时。
4. 3.4 Spatial separation and structural specificity：横向三线、轴向点/线对及断裂/连接判据；实测分辨率方案待补。
5. 3.5 Volumetric fidelity, repeatability, and reporting：体误差、轴向Wasserstein距离、横向支撑外强度、SSIM、子集波动及样本层级。

## 术语与编号

| 项目 | 本节采用的定义 |
|---|---|
| mean-anchored / Taylor-anchored network | 保持既有输入分支和优化日程，区分输出锚点及其对应标定。 |
| Mean-RL3 / Taylor-RL3-sqrt | 保存记录中的名称；实际初始更新式见第2节式(5)，不重新称为标准RL更新。 |
| frame-reduction factor | 只有两种配置达到同一预先确定的质量目标时，才使用基线帧数/网络帧数。 |
| bar width and equal-width gap | 同时报告条宽和等宽间隙；不替代FWHM或光学截止频率。 |
| axial separation | 以原生层上的峰、匹配及谷值判据定义；10 µm相邻层没有中间样本，谷值分离记为不适用。 |
| NRMSE_shape | 仅拟合全局非负强度比例，不执行空间配准。 |
| background mass fraction | 横向支撑掩膜外、所有层累计的强度比例，不是三维体素级真背景。 |
| SSIM | 当前自定义实现：11×11均匀窗、零填充、两幅图合并强度范围、不做gain alignment；不冒充原文Gaussian窗默认配置。 |
| subset repeatability | 同一对象不同测量子集的波动；不是独立对象/动物的重复。 |

公式接续第2节为(16)–(20)。Table 1为当前重写稿首张编号数据表。参考文献[12]新增SSIM原始研究；原Introduction的[1]–[11]和第2节引用无需改号。

## 已核对的事实与来源

| 事实 | 证据 | 状态 |
|---|---|---|
| 11/3/3对象；每对象10/90划分及十组子集 | 指定运行run_contract.json、frozen dataset及输入读取器 | 已核对。 |
| 五个平面训练对象、六个体训练对象 | P01–P11/prepared.mat中的cfg和truth_depth/illumination_model | 已核对；没有把所有训练对象说成同一种体生成流程。 |
| T02中断管状、T03三线、T04轴向薄线对及单层对照 | 冻结geometry.json与prepared.mat | 已核对；当前源代码中其它版本的T04网格对象没有被混入。 |
| 488 nm、照明NA0.05、520裁剪260、1.125 µm照明采样 | 冻结prepared.mat与cell_generate_illumination_3d.m | 已核对；体样本使用共享复场传播。 |
| 物体采样1.12244898 µm、检测NA0.15、z=10:10:100 | 冻结cfg、PSF实际NA/x3objspace字段、指定训练配置 | 已核对；没有从PSF文件名推断z间距。 |
| 当前读取sensor_pre_detector，noise_model=none | 冻结sensor帧及cfg、matlab_multivolume_dataset.py | 已核对；P01上游量化产品例外已说明。 |
| 两网络训练/评价核心输入一致 | preflight.json的baseline_cache_comparison：170实例、2270数组/字段、0不一致 | 依照已保存预检记录；本次不重新扫描25GB数据。 |
| 10%阈值、两像素/一层匹配、0.8谷值、central60% | v3_compare_local_audit.py与保存的supplemental表 | 已核对最终局部评价口径。 |
| NRMSE、W1、SSIM、支撑外强度与子集dispersion | reconstruction_metrics.py、v3_compare_evaluation.py、mean_anchor_analysis.py | 已核对。 |
| 四种现有对照 | analysis/quality_per_subset.csv的120行，4方法×3对象×10子集 | 已核对方法清单；不把待补基线写成已有结果。 |
| 菠菜根探索性结果没有独立GT | 指定运行spinach_root_transfer/REPORT_ZH.md | 保留探索性质，不替代正式生物验证。 |

## 为什么这样组织

- 先交代光学测量及数据来源，使比较条件可理解；再按减帧、细节和伪结构定义评价。
- 用局部结构判据补充全局体误差，避免只用PSNR/SSIM支撑高分辨主张。
- 将现有运行与计划实验通过时态和显式标签区分，同时保留菠菜根和鼠脑切片的完整论文范围。
- 将复现细节留在本节和本说明中，数值性能留给第4节Results，不重复第2节的网络推导。

## 待补证据与建议归位

- 主结果：质量匹配的十帧/高帧比较、实测分辨率、两类生物样本及独立参考。
- 必要支持：合理迭代数传统基线、适配同一测量模型的独立学习基线、与设计主张对应的消融。
- 鲁棒性：相关长度/噪声/PSF失配、多个训练种子、阈值敏感性。若改变结论方向或适用边界，必须在Results主文明确；其他细节可放补充材料。
- 复现记录：相机与物镜等实际型号、曝光/切换/读出、标定与预处理、软件版本和GPU型号、数据/权重/代码位置。
- 样本层级：独立根或动物/供体为生物重复；切片、视野和子集为嵌套测量。未填写样本数、随机化/盲法或伦理批准，均须来自实际记录。
- 既有T02–T04已参与开发比较；后续锁定方法与判据后的独立对象/采集用于进一步评价，不能回溯声称既有数据为最终盲测。

## 文献核对

新增SSIM文献的原作者页面和论文已读取；Crossref核对了题名、作者顺序、卷13、期4、页600–612、2004年和DOI。Crossref将首作者整个姓名放在family字段，本稿采用原作者页面确认的Z. Wang，不照抄该元数据拆分错误。

- 原作者页面：https://ece.uwaterloo.ca/~z70wang/publications/ssim.html
- 原作者PDF：https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf
- DOI：https://doi.org/10.1109/TIP.2003.819861
- OE审稿标准：https://opg.optica.org/resources/author/Optics_Express_Research_Article_criteria_Oct2025.pdf

## 全文衔接

第2节2.5已有数据数量与训练设置。本节为独立可审阅稿保留必要数量；全文合并时，可保留本节的数据设计，压缩第2节中的重复对象数量句。当前不修改已交付的Introduction和Principle and method。

后续可按“小节号、具体句子或评价判据”反馈，以便针对性修改。
