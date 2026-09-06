# MATLAB 五样本试验数据集

该目录实现已经确认的第一阶段数据协议。目标是先验证 5 个不同三维样品，
不把 AlgoRIM 输出当作 GT，也不向未知深度重建泄漏真实层位置。

## 固定样本

| ID | 物体 | 真实深度 |
|---|---|---:|
| P01 | 现有 USAF / 星形图 | 50 µm |
| P02 | 线对目标 | 20 µm |
| P03 | 稀疏点与短细丝 | 80 µm |
| P04 | 弯曲、交叉、断裂细丝网络 | 30 µm |
| P05 | 弱环与膜状结构 | 100 µm |

每个样本 100 帧；重建深度固定为 `10:10:100 µm`。十个确定性子集各取
10 帧作为网络输入，其余 90 帧只作为约束。十组输入恰好划分 1–100 帧。

## 数值口径

- 散斑生成保留原 `generate_speckle_NA05.m` 的相位、FFT、pupil、裁剪和逐帧最大值流程。
- P01 的 PNG 已经是 `object × speckle`，绝不再次乘样品。
- 检测器前向使用原 `forwardProjectACC`。
- 传统重建使用原 `forwardProjectGPU`、`backwardProjectGPU` 和 `deconvRL`，固定 3 次迭代。
- mean 使用完整十层 `H/Ht`；Taylor 使用完整十层 `H.^2/Ht.^2`。
- 统计量从 TIFF 编码前的浮点帧计算，方差固定为 `var(...,0,3)`，即 N-1 归一化。
- mean 与 var 都来自同一组 10 帧，不添加 uniform exposure。
- PSF 深度只按 `x3objspace` 匹配；不采用文件名或冲突的 `zspacing=14.6154 µm`。
- GPU 常驻 PSF 只消除重复主机—显存传输。3-step A/B 为逐元素完全相同。
- 单精度 FFT/RL 的负残差按峰值幅度和总负质量双门槛处理，详见
  `NUMERICAL_ROUNDOFF.md`；任一门槛超限即停止，不发布结果。

## 运行

MATLAB 中执行：

```matlab
addpath(genpath('matlab_code'));
report = run_matlab_pilot({'P01'}, '', 6);
```

第三个参数是空闲 GPU 数量。运行前必须确认没有占用他人的卡。完成文件可安全续跑；
同版本但不完整的临时文件不会被发布。

## 主要输出

默认根目录：`data/matlab_pilot5_v1/<sample>/`

- `prepared.mat`：冻结配置、样品、十层 GT 和 10/90 索引。
- `sensor_frames/`：100 张保留尺度及严格旧版检测帧。
- `recon_frames/`：100 个逐帧十层重建体。
- `subsets/`：10 个网络输入/90 帧约束统计与四类十层传统重建。
- `algorim/`：Windows AlgoRIM 的 20 个深度栈及尺度清单。
- `previews/`：GT、子集十页 TIFF 和汇总图。
- `validation_manifest.mat/json`：全量验收和 SHA-256 文件清单。
- 根目录 `pilot5_subset_01_comparison.png`：五样本真实层图像对比。
- 根目录 `pilot5_all_subset_axial_metrics.csv` 和
  `pilot5_axial_summary.json`：全部 200 条/20 组轴向指标汇总。

## 最终完成状态

2026-09-05 已完成并通过绘图后二次验证：

- P01–P05 均包含 100 个逐帧重建、10 个固定子集、40 个十页子集预览，
  以及 20 个 100 页、16-bit AlgoRIM 栈。
- P01 清单包含 377 个已哈希产物；P02–P05 各包含 478 个。P01 复用历史
  `object × speckle` 输入，因此没有 P02–P05 的独立 illumination/product 中间文件。
- MATLAB 回归测试 21/21 通过，失败 0，未完成 0。
- 独立 TIFF 读取器再次核验了每个样本的 100 张 detector TIFF、40 个十页
  subset TIFF 和 20 个百页 AlgoRIM TIFF。
- 数据集总大小约 6.3 GiB。

传统 3-step mean/Taylor 体是网络输入和 AlgoRIM 辅助数据，不是 GT。其轴向质量
本身较宽，不能把传统体的峰值层命中率当作数据集正确性的验收条件；定量真值只来自
`prepared.mat` 和 `ground_truth_float.tif` 中声明的唯一真实层。
