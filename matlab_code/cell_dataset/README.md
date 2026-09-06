# 连续三维细胞数据集：MATLAB 真值预览与完整仿真

这里保留两个彼此隔离的入口：`RUN_CELL_DATASET_PREVIEW.m` 只生成真值并停下等待
人工确认；`RUN_FULL_MATLAB_DATASET.m` 只接受已经冻结且获批的真值，继续执行散斑
成像、RL、10/90 子集和 AlgoRIM 输入栈导出。原 `matlab_pilot5_v1` 的所有文件保持原样。

r04 完整仿真已完成。权威状态见
`data/matlab_cells_pilot_v2_r04/FINAL_DATASET_MANIFEST.json`；数据划分状态见同目录
`dataset_splits.json`。服务器生成的是 Windows AlgoRIM 的输入栈，没有冒充已经运行过
AlgoRIM 应用程序。

## 一键运行

在 MATLAB 打开项目根目录的 `RUN_CELL_DATASET_PREVIEW.m`，点击 **Run**，
或切换到项目根目录后执行：

```matlab
RUN_CELL_DATASET_PREVIEW;
```

Linux/Windows 使用同一套 MATLAB 脚本，无需 Python。默认输出到
`data/matlab_cells_pilot_v2_r04`。旧预览（包括 r03 珠串/密网版本）全部保留。
无需 GPU，但 MATLAB 需要正常许可证和图形导出环境。
当前服务器的 MATLAB 是 R2023b；没有在 Windows 实机验证。

指定独立输出位置：

```matlab
RUN_CELL_DATASET_PREVIEW(fullfile(pwd,'data','my_cells_preview'));
```

生成器源码和参数有指纹检查。脚本未改变时可以再次运行重导出预览；改动
形态脚本后请使用新的输出目录，不覆盖已审阅的真值。此版本不加载历史网络 checkpoint。

## 样本和划分

| ID | 三维形态 | 用途 |
|---|---|---|
| P06 | 神经突起样连续分叉、不同深度的交叉 | train |
| P07 | 三个区域的细胞骨架束，含弱细丝和真实断口 | test |
| P08 | 120 个随机散布的实心球；三维互不连接，XY 投影基本分开 | train |
| P09 | 三个区域的短管、弯曲线粒体样结构和局部分支 | validation |
| P10 | 简化的连续三维网状体；网孔更疏、短边合并、弯折减小，保留跨深度连接 | train |

`dataset_splits.json` 同时列出 P01--P06、P08、P10（train），P09、V01、V02
（validation），以及 P07、T01、T02（test），总计 8/3/3 个物体。P01--P05
已经复制到 r04 根目录，因此最终 14 个对象都在同一个数据集目录下。
不能把同一物体的不同 10 帧子集分给不同 split。

四个新的独立单层对照也已留下构建函数，可单独生成真值预览：

```matlab
addpath(fullfile(pwd,'matlab_code','cell_dataset'));
run_cell_truth_preview('',{'V01','V02','T01','T02'});
```

## 物理尺度和真实程度

- 所有控制点、曲线宽度和曲面参数以微米定义，不以像素假扮生物尺寸。
- XY 为 260×260，采样 `220/49/4 = 1.12244898 um`。
- 连续曲线按最大约 0.7 um 的弧长间隔离散；横截面为 3σ 截断高斯。
- 为避免细结构恰好落在两个 Z 平面之间而消失，几何真值以 1 um Z 间隔采样。
- 正式十层 GT 是每个中心位于 10:10:100 um、厚度 10 um 的 slab 内平均密度。
  这是可记录的离散化约定，不是新的细 Z PSF，也不意味着系统有 1 um 轴向分辨率。
- 只有一次全体积最大值归一化。层质量由几何自然决定，不进行逐层归一化。
- fine/coarse 数值共用同一归一化；`sum(fine)*1` 与 `sum(coarse)*10` 保持一致。
- 这些是尺度适配的单通道荧光形态 phantom，不是纳米级细胞器真值或真实显微图。
- maximum union 合并相交的几何支撑，避免同一连续分支的离散线段叠加形成假亮点。
- P08 共 120 球，半径 5.8--7.8 um，内部均匀填充，边界使用 0.6 um 半宽平滑过渡。
  球心 XYZ 随机提议后按间距筛选，不使用曲线、网格或环；XY 球心范围 22--268 um，
  Z 球心范围 28--82 um。包含平滑边缘在内的三维球表面间距至少 3 um，无连接管。
  前 108 球在 XY 投影也留出至少 2 um 间隙；后 12 球仅允许少量轻微投影交叠，
  三维仍不接触。投影接近不代表真实连接。数量和间距可在配置的 `beads` 中修改。
- P10 用三维空间内独立分布、保留最小间距的随机种子生成真正的三维 Voronoi 边网。
  体内随机种子从 420 减至 320，最小间距从 12 增至 16 um，合并短于 2.5 um 的边，
  中点弯折幅度从 3 降至 0.8 um；管宽 sigma 仍为 0.8--1.15 um。
  整体保持一个连通图，中心线 Z 约 22--86 um；不通过减少体厚或改成二维网片简化外观。
  实际边数、合并数写入 geometry.json，不把预设目标数当成实测结果。

## 每个样本的产物

```text
P06/
  truth.mat                        cfg、fine/coarse GT、几何参数、索引、指纹
  ground_truth_float.tif            10 页 single；页面是 Z，不是时间
  ground_truth_fine_float.tif       100 页 single；1 um 几何 Z，不是散斑帧
  geometry.json                    曲线控制点、曲面参数、归一化语义
  truth_metrics.json               层质量、质心、质量守恒检查
  previews/
    layers_native.png              十层逐层图，共享一个显示范围
    projections_physical.png       fine/coarse 的 XY、XZ、YZ 最大值投影
    view3d_physical.png             两个视角，真实 XYZ 微米比例
    view3d_physical.fig             MATLAB 内可旋转的三维视图
    axial_mass.png                 真值层质量
```

根目录的 `overview_P06_P07_P08_P09_P10.png` 是五样本总览。
`MORPHOLOGY_REVIEW_REQUIRED.json` 明确记录 approved=false。该文件不是授予自动执行的凭证；
人工同意形态之后仍须单独调用经验证的下一阶段程序。

注意：`ground_truth_fine_float.tif` 虽有 100 页，但绝不是 AlgoRIM 的 100 帧时间栈。
本阶段没有 `sensor_frames`、`recon_frames`、`subsets` 重建结果或 `algorim` 栈。
MAT 内的 input/holdout indices 只是提前冻结的帧划分，不表示采集帧已经生成。

Python 读取轴序要显式转换：MATLAB 数组为 YXZ，h5py 读取 v7.3 MAT 后为 ZXY，
应使用 `array.transpose(0,2,1)` 得到 ZYX；tifffile 读取这里的 TIFF 已是 ZYX。
XY 尺寸都是 260，漏掉这个转换不会报尺寸错误，却会把图像转置。

## 如何修改形态

修改 `cell_make_truth.m` 对应 P06--P10 的 case：

- `addtube(control_xyz_um, sigma_um, amplitude)`：控制点定义三维曲线；FWHM 约 2.355σ。
- `addsolidball(...)`：实心球，中心位置、半径、亮度与球体编号独立记录。
- `addshell(center_xyz_um, radii_xyz_um, sigma_um, amplitude, phase)`：空心膜壳。
- `addsheet(...)`：带孔的倾斜/起伏片层。
- `cell_dataset_config.m` 的 geometry_seed：同一规则下改变细部几何/亮度。

改变 seed 并不保证所有样本控制点变化：P06 主干仍是固定设计，
若要独立新物体也应改变主干路径；P08 位置与 P10 网络拓扑则由 seed 随机生成。

## 验证

```matlab
addpath(fullfile(pwd,'matlab_code','cell_dataset'));
results = runtests('matlab_code/cell_dataset/test_cell_truth.m');
assertSuccess(results);
```

覆盖五个连续体、四个单层对照、数值和尺寸、层质量守恒、重复生成一致、
随机状态隔离、物体级划分，以及拒绝逐层归一化。r04 另检查 120 球内部填满、
三维表面间距、投影间距、分布范围、P10 短边合并后全网连通及跨深度连接，
并验证 P06/P07/P09 与已审阅版本真值逐元素不变。

## 人工确认后的后续阶段（本次不执行）

保留原 MATLAB forwardProjectACC / RL 核心，另接入三维照明与深度叠加。
原散斑生成器只产生二维强度；跨深度照明需要保留复场并显式传播，不能复制
强度图后声称真实传播，也不能默认各层独立以迎合 H² 模型。
应先通过参考层一致性、多层前向求和、尺度与数值测试，再生成 100 帧、
3-step mean/Taylor、10/90 子集和 Windows AlgoRIM 栈。

本次交付的“一键”是可复现的五样本几何仿真及预览；完整成像/重建一键入口
需在形态确认后接入和验证，不能把未运行的流程宣称为已完成。

## r04 形态确认后的完整仿真

形态已于 2026-09-05 确认。完整数据集入口为项目根目录的
`RUN_FULL_MATLAB_DATASET.m`：

```matlab
cd('/workspace/xyx/speckle_illumination_net');
RUN_FULL_MATLAB_DATASET(1:6);
```

六个独立 MATLAB 进程分别固定到六张 GPU，并行处理不同样本；每个样本内部按帧
顺序保存。已有且协议匹配的完整阶段会复用，因此相同命令可以在中断后继续。
新样本使用同一复散斑场的角谱传播构造十个深度照明，绝不逐层独立抽样或逐层归一化。
生成 100 帧、逐帧三维 RL、10 组 10/90 统计与四类 3-step 初始重建、以及 Windows
AlgoRIM 所需的 20 个 100 页栈。服务器不会运行 Windows AlgoRIM 软件本身。
