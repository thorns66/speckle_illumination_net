# N=200/400/800：固定 50 um 完整 Cs 方差图诊断

日期：2026-09-04

## 决策

六组实验全部完成 200 step，均使用空闲 A40 并行运行。完整 Cs 方差图匹配在各帧数下都使
FTC 和真值 L2 小幅改善，但没有同时改善毛刺、峰值和相位指标，因此不接入正式 no_mean。

这一结果修正了“N=100 统计不足是主要瓶颈”的单一解释：N=800 时收益仍然有限，不能继续
只靠增加帧数或提高损失权重。下一步应区分“只用传感器协方差对角线造成的信息不足”与
“16 个模型探针的随机梯度/自噪声偏置”，而不是调整 Set、gate 或深度先验。

## 设计与完整性

- 固定真实 50 um，仅优化二维非负单位质量结构；不测试未知深度，不宣称三维定位通过。
- 一次生成 800 张训练帧、800 张独立 holdout 帧，再取前 200/400/800 张形成嵌套数据集。
- 数据 seed 20260940；用于系统尺度和 Cs 的 2048 个散斑与训练/holdout 分离。
- 三个数据集使用同一系统尺度、Cs、真值及 LFM，未逐帧归一化、量化或添加相机噪声。
- 每个 N 只用自己的训练帧重新计算 RL3 anchor。嵌套前缀已逐元素一致性检查；全部有限。
- N=200/400/800 分别测试 B=0.5/2，学习率 0.01，随机种子 20260901，严格 step 0--199。
- 训练 16 个 Gaussian Cs 探针每步刷新；holdout 使用固定独立 512 探针；sigma=0.5 px。
- 选择标准为同 N 独立经验 holdout 方差图损失；真值与 FTC 从不参与 checkpoint 选择。
- 六组均正常退出，峰值 PyTorch 显存约 24,950 MiB；各组优化及验证约 913 秒。

此处使用完整 Cs 计算传感器方差对角线，不是完整 sensor covariance-vector matching。

## 主结果：按 holdout 选出的 B=0.5

| N | anchor FTC / um | Cs FTC / um | anchor L2 | Cs L2 | anchor 峰值比 | Cs 峰值比 |
|---|---:|---:|---:|---:|---:|---:|
| 200 | 7.592 | 7.135 | 0.30979 | 0.29935 | 1.358 | 1.443 |
| 400 | 7.500 | 7.043 | 0.30430 | 0.29067 | 1.268 | 1.459 |
| 800 | 7.500 | 7.135 | 0.30138 | 0.28675 | 1.168 | 1.319 |

FTC 改善约 4.9%--6.1%，真值 L2 改善约 3.4%--4.9%。但 B=0.5 的归一化 Laplacian
比对应 anchor 增加约 12%--13%，目标相位一致性下降约 0.012--0.014，环带梯度余弦也下降。
背景泄漏减少，但不能用这一项抵消细节方向/峰值质量退化。

B=2 的 best FTC 与 B=0.5 完全相同，L2 仅再改善 0.0004--0.0006，不能证明增加自由度值得。

## best 与 final

六组 holdout best 都位于 step 139；final 为 step 199。Final FTC 为：

- N=200：7.043 um；
- N=400/800：6.952 um。

Final 的真值 L2 比 best 稍好，但 holdout 更差。不能用真值把 final 替换成主结果。
所有组在同一步获得 best，且共享相同模型探针序列，提示下一步需要审计模型 Monte Carlo
噪声及 checkpoint 选择稳定性；这只是线索，尚不能据此认定因果。

## 视觉核对

固定同一显示范围检查左上角：条纹间隙略清楚，但细小亮暗不均匀和局部峰值仍增强。
没有达到“分辨率更高且更干净”的联合目标。

统一评价使用真实星靶校正后的固定中心 (10,12)（2 倍评价网格），毛刺环带为 20--120 px。
评价脚本已对旧 N=100 anchor 回归，FTC=7.317483、L2=0.321031、radial=0.186241、
Laplacian=0.157419，均与上一轮报告一致。

## 下一步的区分性验证

1. 在同一 N=800 数据上，用确定性 Cs FFT action 构造 sensor covariance-vector matching，
   检查加入非对角信息是否接近 oracle；不继续只优化方差对角线。
2. 对当前方差图目标单独检查 16/128 训练探针和双独立探针交叉估计器，并用第三组大探针数
   重算 best/final。这样才可区分信息不足与模型估计噪声，而不是混在一起调参数。
3. 仍只做固定真实深度、200 step；只有 FTC、真值 L2、相位、峰值和毛刺联合通过，才恢复
   未知深度/三维实验。

本轮仅一个独立数据 seed，因此不能从这张表宣称真实采集至少需要 200、400 或 800 帧。
正式 no_mean、Set branch、gate、PSF 和生产配置均未因本轮修改。

## 产物与元数据注意

- 数据：`outputs/linear_float_oracle_50um/frame_count_nested/n200|n400|n800/`
- 六组结果：`outputs/linear_float_oracle_50um/frame_count_results/`
- 统一表：`frame_count_summary.json`、`frame_count_summary.csv`
- 视觉对比：`upper_left_comparison.png`
- 实际生成脚本：`tools/generate_nested_frame_count_dataset_v2.py`
- 并行入口：`tools/run_nested_frame_count_cs6.sh`
- 统一评价：`tools/summarize_nested_frame_count_cs.py`

复用的临时训练原型在每组 summary 的两个描述字符串中硬编码了 N=100；这只是标签错误，
实际输入和计算使用完整的各组 200/400/800 帧。以 dataset/metadata.json、arguments.dataset
及统一 frame_count_summary 的 frames 字段为准。补丁工具的路径映射问题导致原文件修改失败，
故此处显式记录，不将错误标签作为实验事实。
