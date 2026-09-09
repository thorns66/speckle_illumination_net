# 当前网络架构与训练实验思路

> **当前基线已更新为 E3＋100% mean 结构梯度上限，默认使用第 400 步 final。** 详见 [当前基线记录](CURRENT_BASELINE.md) 和 [模型及来源清单](CURRENT_BASELINE.json)。下文保留历史架构说明；其中旧的“当前基线/正式基线”表述以该记录为准。

整理日期：2026-09-05。依据当前工作区源码、正式 no_mean 配置和 MATLAB pilot5 数据协议整理。本文件是方案与代码状态快照，不表示启动了新训练，也不修改已有实验。

## 1. 核心结论与状态边界

目标是用更少的散斑帧得到更好的三维重建，重点改善横向分辨率，同时避免毛刺、伪结构和错误深度集中。

当前正式基线是 **VAR + Mean + mean/std SetBranch + Gate，使用 no_mean 方差损失**。`no_mean` 是关闭均值数据项，不是删除 Mean 输入分支。当前 Python 入口仍然对每个物体独立优化网络参数，尚不是通过多样本训练后直接推理的通用模型。

| 内容 | 当前状态 |
|---|---|
| 三分支锚定残差网络、Taylor 方差物理、activation checkpoint | 已接入正式训练 |
| Set Transformer、detail、抗混叠、bounded-log、轴向/横向解耦 | 有候选实现或实验配置，不是正式默认配置 |
| 完整 Cs / covariance-vector matching 等 | 诊断研究路线，不是当前正式训练入口的数据项 |
| 5 个样本、每个 100 帧、10 套固定 10/90 划分、MATLAB 初步重建 | 已生成离线数据 |
| 多物体共享网络训练，10 帧输入、其余 90 帧约束 | 后续拟实现；现有入口没有此训练循环 |
| Windows AlgoRIM 高质量辅助结果 | 已准备输入栈；不能将输入栈当成已完成的 AlgoRIM 输出 |

本次核对发现 README 中“没有 Transformer 路径”的旧描述已经落后于源码：候选实现存在，但正式基线并未启用。

## 2. 当前正式网络的数据流

```text
选定 N 帧传感器图像 y_n
  ├─ 计算均值 mu、样本方差 V（分母 N-1）→ 物理损失目标
  └─ y_n - mu → 共享 2D CNN → mean/std 集合聚合 → Z lifting → S0,S1,S2

F_var 三维方差重建 → RMS 特征归一化 → 3D VAR Encoder  → V0,V1,V2
g_mean 三维均值重建 → RMS 特征归一化 → 3D Mean Encoder → M0,M1,M2

每个尺度：(V_l,M_l) → 物理特征融合，再加 alpha_l × gate_l × Set 修正
  → 三尺度 3D Decoder → 有正有负的残差 R
  → 与 A = beta × F_var 做非负锚定融合 → 最终三维重建 g
  → H(g)、H2(g²) → 记录均值误差，优化方差误差和弱 TV
```

物理前向不在网络内部学习 PSF。Set 输入是去均值后的传感器帧，不是每帧 RL 三维体；若未来改成后者，需要单独设计输入接口和消融，不能当作现有架构原样使用。

### 2.1 输入与尺寸

| 输入 | PyTorch 尺寸 | 意义 |
|---|---|---|
| `f_var` | `[B,1,Z,H,W]` | 非负 VAR 幅度体，也是输出锚点 |
| `g_mean` | `[B,1,Z,H,W]` | 辅助强度、支持和深度特征 |
| `residual_frames` | `[B,N,1,H,W]` | 所选帧减去它们自己的均值 |
| `z_values_um` | `[Z]` 或 `[B,Z]` | 重建网格坐标，不是样品真实深度标签 |

正式单体优化中 B=1；已有 50 µm 数据通常 Z=10、H=W=260、N=100。重建范围为 10–100 µm，每 10 µm 一层。

VAR/Mean 的 RMS 归一化只用于编码器特征：`x / sqrt(mean(x²))`，尺度由 detached 输入计算并设下限。输出锚点和物理损失不使用这套特征归一化。Set 输入没有额外逐帧 RMS 归一化。

### 2.2 VAR 与 Mean 编码器

两路使用独立参数、相同的三尺度 3D CNN 结构：通道 `[16,32,64]`。

每个 ConvBlock 包含两次 `3×3×3 Conv → GroupNorm → SiLU`。尺度间以 stride `(1,2,2)` 的卷积降采样，只降低 XY 尺寸，不降低 Z 层数；3D 卷积仍然允许跨层特征交互。

在 260×260 输入、无额外 context 的基线下，三层输出分别为：

| 尺度 | VAR / Mean 输出 |
|---|---|
| 0 | `[B,16,10,260,260]` |
| 1 | `[B,32,10,130,130]` |
| 2 | `[B,64,10,65,65]` |

### 2.3 原始 SetBranch

1. 每张去均值帧通过同一个共享 2D CNN，通道 `[8,16,32]`。
2. 各尺度沿帧维计算特征均值与标准差。这里的特征方差使用除以 N 的二阶矩，不要与传感器样本方差的 N-1 分母混淆。
3. 拼接 mean/std，经 `1×1 Conv2d` 压回该尺度通道数。
4. 将二维特征复制到各 Z 层，拼接 `z_um / 100` 坐标通道，经 3D ConvBlock 得到三维 Set 特征。

该聚合不依赖帧排列，不知道每帧真实散斑。它提供统计特征修正，不直接保证恢复某个高频或提高分辨率。默认每次处理 8 帧，并对共享 CNN 使用 activation checkpoint。

### 2.4 Gate 融合

每个尺度 l：

\[
P_l=\operatorname{Conv}_{1\times1\times1}([V_l,M_l]),\qquad
G_l=\operatorname{sigmoid}(\operatorname{Conv}_{1\times1\times1}([P_l,S_l]))
\]

\[
F_l=P_l+\alpha_l G_l\operatorname{Conv}_{1\times1\times1}(S_l),\qquad
\alpha_l=\operatorname{sigmoid}(\text{raw\_alpha}_l).
\]

Gate 是每个体素一个标量，在通道间共享；三个 alpha 是三个尺度的全局可训练标量，初始化约 0.05。它们控制 Set 修正的注入量，不是最终图像。

`gate_scale{l}_mean.png` 是 gate 数组在非 XY 维度上平均后的显示图，不是 Mean 分支重建，也不是 Set 分支输出的高分辨率图。不能凭 gate 图是否清晰判断模块是否有效，应依据重建消融结果。

### 2.5 Decoder 和非负输出

Decoder 在最深尺度处理特征，然后两次三线性上采样到对应 skip 尺寸，拼接并卷积，最后 `1×1×1 Conv` 输出单通道残差 R。最后一层权重和偏置均零初始化。

基线设 A=beta·F_var，并使用：

\[
g=\begin{cases}
A+R,&R\ge0,\\
A\exp(R/\max(A,\epsilon)),&R<0.
\end{cases}
\]

因此初始输出恰好是 A；正残差可以增加结构，负残差平滑减弱锚点且保持非负。这不是最终 V2 中的单位质量参数化，也不是默认轴向 softmax 模型。

全局 beta 的初始化与约束：

\[
\beta_0=\max\left(\frac{\langle H(F_{var}),\mu\rangle}{\|H(F_{var})\|^2+\epsilon},\epsilon\right),\qquad
\beta=\beta_0[1+0.2\tanh(\text{raw\_beta})].
\]

beta0 在开始时解析初始化；raw_beta 随方差目标一起优化。不是每一步重新解析计算 detached 全局增益，也没有逐层 beta_z。

## 3. 正式物理与训练目标

### 3.1 前向物理

\[
\hat\mu=H(g),\qquad
\hat V=H_2(g^2)+\alpha_{noise}\mu+\sigma_{read}^2.
\]

H2 指对每个 PSF 核逐元素平方后形成的前向算子，不是连续应用两次 H。当前 50 µm 仿真配置的两个噪声参数均为 0。

H 是周期位置相关的 LFM 算子：每层有 49×49 个相位位置对应的核；49×49 不是微透镜总数量。Python PSF 轴序为 `[Z,phase_y,phase_x,kernel_y,kernel_x]`。正式算子保留相位相关性，不把它简化成一个普通卷积核。

此 Taylor 模型没有完整有限宽度 Cs 的交叉项。它是当前回退基线，不应描述成已经准确解决了真实跨深度散斑协方差。

### 3.2 实际优化的损失

\[
L_{var}=\operatorname{SmoothL1}(\log(\hat V+10^{-6}),\log(V+10^{-6})),
\]

\[
L=L_{var}+10^{-5}\,[\operatorname{mean}|D_xg|+\operatorname{mean}|D_yg|+0.5\operatorname{mean}|D_zg|].
\]

这是未减去空间均值的 log 方差损失，保留当前数据尺度下的幅度差异；不能称为中心化 shape-only 损失。日志中的 `normalized_var_loss` 就是上述 log 损失，而 `raw_var_loss` 是未取 log 的 SmoothL1，仅作记录。

均值误差继续计算和记录，但 `lambda_mean=0`，不通过均值数据项更新网络。均值仍通过输入特征和 beta 初始化发挥作用，所以“完全不用均值信息”也不准确。

### 3.3 优化、步数与输出

| 项目 | 当前正式配置 |
|---|---|
| 优化器 | Adam，两组参数 |
| 网络学习率 | 1e-3 |
| raw_beta 学习率 | 1e-4 |
| 随机种子 | 20260901 |
| AMP | 关闭 |
| PSF phase chunk | 32 |
| 物理 activation checkpoint | 开启 |
| 保存间隔 | 20 步，并保存最后一步 |
| best 选择 | 最小训练 total loss，不用真实深度或分辨率挑 checkpoint |

重要差异：当前 `configs/depth50_n100_no_mean_loss.yaml` 仍写 `max_steps: 400`；用户最新约定是后续测试 200 步。本次仅整理，不改配置。执行新测试应显式指定 `--max-steps 200` 和独立输出目录，避免覆盖旧基线：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
python -u train_volume.py \
  --config configs/depth50_n100_no_mean_loss.yaml \
  --device cuda:0 --init random --max-steps 200 \
  --output-dir outputs/no_mean_200_new_run
```

该命令只是旧单体入口示例，不是新 pilot5 的多样本训练命令。200 步循环编号为 0–199；重建快照和 best 是对应步更新前前向计算的输出。主要产物为 `reconstruction_best.tif/.npy`、`checkpoint_best.pt`、逐步重建、`losses.csv`、`config_used.yaml` 和 gate 可视化。比较时同时报告 best 与 step 199。

## 4. 现有实验思路与解释边界

### 4.1 分支消融

保持输入数据、帧数、损失、随机种子、步数及评价方式一致：

| 组别 | VAR | Mean | Set |
|---|---|---|---|
| E0 正式基线 | 开 | 开 | mean/std |
| E1 | 开 | 关 | 关 |
| E2 | 开 | 开 | 关 |
| E3 | 开 | 开 | Set Transformer |

所有组均为 no_mean 目标。当前 Transformer 候选使用共享 CNN、两层 ISAB、4 heads、16 inducing points 和 PMA，学习全局帧权重后做加权空间 mean/std 聚合，不是对整个三维体直接做全体素注意力。

已有 400 步分支报告没有证明 E3 全面超过 E0；E2 的三种子中有一次峰值落到 40 µm。因此不能把“Transformer 已实现”当作“已验证能提高分辨率”。历史 400 步报告不能直接当成新的 200 步公平对照。

### 4.2 已尝试/候选路线

代码保留了 context padding、抗混叠、detail head、频带损失、bounded-log 与轴向/层内解耦等研究接口。完整 Cs、协方差 sketch 和固定真实层二维优化主要见诊断工具与报告。

这些接口或固定层诊断的存在不等于正式方案通过验收。固定 50 µm 的结果只反映“已知深度时的横向优化潜力”，不能作为未知深度、任意三维体的重建成功证据。本文件不据此重新启用已经失败的路线。

### 4.3 评价不能只看 loss 或一个 FTC 数字

共同记录：真实层图像、原始 VAR、传统结果和网络结果；固定显示范围、裁剪、中心和采样条件；再比较 FTC、相位一致性、中心漂移、毛刺/Laplacian 指标、最大体素占比和完整轴向分布。

对星形分辨率样本，固定参考中心 FTC 作为主要可比结果，注册中心结果辅助解释位移，不能用独立中心注册掩盖几何偏移。旧固定错误中心得到的 VAR=13.903 µm、no_mean=7.500 µm 已在评价说明中判为无效，不能复用作结论。

单层目标同时检查峰值层、真实深度±10 µm 质量、质心、边界质量；多层目标改用层质量误差、轴向 W1、支持外伪质量和横向相似性。禁止为了获得漂亮横向指标而指定真实层参与优化。

六张 A40 可将独立配置/随机种子分配到空闲卡并行运行；不抢占 FDTD 等现有任务。这与把一个模型做六卡分布式训练是两回事。

## 5. 新 MATLAB 数据集与网络的接点

数据位置：`data/matlab_pilot5_v1`。P01/P02/P03/P04/P05 的真实深度分别为 50/20/80/30/100 µm，每个 100 帧、10 层、260×260，当前是无额外相机噪声的单层仿真。

每个样本有 10 套固定划分，每套 10 帧输入，另 90 帧互斥留出。10 个输入子集之间也互不重叠，但其 90 帧补集高度重叠；它们不是 10 个独立物体。

| 离线产物 | 含义与网络接入建议 |
|---|---|
| `subsets/subset_XX.mat: physics_mean_raw` | 对输入 10 帧的均值做 3 次 RL；作为 Mean 体的候选浮点输入 |
| `physics_taylor_raw` | 对输入 10 帧的方差做 3 次平方 PSF RL；尚未开方 |
| `physics_taylor_sqrt_float` | 上述结果开方一次；对应 VAR 幅度锚点候选 |
| `input_indices` | 从传感器数据中读取同一组 10 帧，构造 Set 输入 |
| `holdout_*_mean_float` / `holdout_*_variance_nminus1_float` | 90 帧的统计约束目标，不进入 10 帧输入编码器 |
| `recon_frames/frame_XXX.mat` | 每帧独立的 3D RL 结果，主要供 AlgoRIM 同层堆栈；不是当前 Set 默认输入 |
| `algorim/*/depth_XXXum.tif` | 每个深度 100 帧、16-bit 输入栈，等待 Windows AlgoRIM 处理 |
| `prepared.mat: ground_truth` | 仿真真值；与传统重建输出严格区分 |

这里的映射是下一步数据接口设计依据，尚未由 `train_volume.py` 自动读取这些 MAT 字段。现有入口读取 TIFF，对整数按 dtype 最大值缩放；不能直接将显示归一化的 TIFF 与保留浮点尺度的 MAT 混用。

特别注意历史基线的 `g_mean_tiff_path` 指向名称含 `uniform` 的旧文件；新数据协议已经改为同一输入 10 帧的均值，不允许额外 uniform 曝光。迁移时应更换数据接口并单独记录这种输入变化，不能只修改 N。

### 5.1 尺度与光学参数的风险

`legacy` 与 `physics` 是两种数值处理版本，不是两组独立观测。前者兼容旧流程逐帧归一化/量化；后者尽可能保留相对尺度。P01 原始 object×speckle 已是 8-bit，无法恢复之前丢失的绝对照明尺度。

MATLAB 当前冻结协议保留生成器参数 NA=0.05、波长 488 nm、散斑网格 1.125 µm；LFM PSF 文件标称波长 532 nm，物方采样约 1.12244898 µm。不能把早期讨论中的 NA=0.04479、532 nm 参数自动当成这批生成散斑的实际真值。若后续使用完整 Cs，必须先核对这些参数和输入 TIFF 的适用性；不在整理阶段悄悄重采样或修改原 MATLAB 流程。

## 6. 下一阶段：多样本学习的可行训练思路（尚未实施）

### 6.1 与现有逐样本优化的本质区别

现有模式：一个物体 → 初始化网络 → 对该物体优化 200 步 → 输出该物体。

拟采用模式：多个训练物体与其 10 帧子集 → 更新同一套共享网络参数 → 新物体仅给 10 帧及其 MATLAB 初步重建 → 前向预测三维体。是否再做少量自适应微调，应作为单独实验，不能混同直接推理结果。

通用网络仍优先保持 VAR 主分支、Mean 辅助分支和 Set 统计修正结构，先验证训练范式与输入是否有效，不同时替换 gate、PSF、输出参数化和多个损失。

### 6.2 推荐按以下顺序接入

1. **完成数据接口对齐。** 使用已有固定 10 帧子集的 mean、VAR 和传感器帧，核对尺度、轴序、帧编号与 3 次 RL 输出。MATLAB YXZ 与 Python ZYX 的转换要显式测试；MATLAB 帧号从 1 开始，Python 从 0 开始。
2. **建立 10 帧输入、90 帧训练约束版本。** 预测体送入物理算子，与该划分剩余 90 帧的统计量比较。先保留 no_mean 方差目标和弱 TV；均值误差作为监控。当前物理近似是否匹配仍需无网络残差检查，不能仅因数据量增加就假定 H2 模型变准确。
3. **先解决 beta 的跨样本定义。** 当前 raw_beta 是逐物体参数，不能不加处理地共享给所有样本。一个可测试的方案是每样本用输入 10 帧均值计算 beta0，再决定固定 beta0 或学习共享的受限修正。该选择尚未确定，须通过尺度检查和消融验证；不能用 90 帧均值初始化推理输入增益。
4. **把 AlgoRIM 作为可选辅助，而非真值替代。** 100 帧 AlgoRIM 图可提供结构参考，但可能有模型偏差。先运行无教师版本，再单独加入教师约束比较。涉及训练对象的 100 帧参考可以用于训练监督，但不能出现在测试对象的输入或调参过程中。
5. **若使用仿真真值训练，单独标记为监督实验。** 真值监督、90 帧统计自监督、AlgoRIM 教师监督是不同方案，必须分开报告，不能统一称为无监督。

这里称作 holdout 的 90 帧，一旦参与梯度更新，就是训练目标，不是独立测试集。最终泛化评价要依赖未参与训练的物体及其独立采集数据。

### 6.3 五个样本阶段可以证明什么

先做数据流、梯度、尺度和单个训练 batch 的冒烟检查，再做按物体划分的小规模验证。不得把同一物体的不同 subset 分别放到训练和测试后声称跨物体泛化。

五个物体每个对应不同深度，图案类型与深度存在混杂，因此只能用于初步可行性检查。即使验证表现好，也不足以证明任意深度、任意三维形态泛化；后续需要同类图案跨深度、多层和厚体样本。

用户约定的 200 步适用于后续单体优化对照；多样本训练应明确“optimizer 更新步数、每步样本数、epoch 数”，不能把 200 次更新直接叫 200 轮全数据训练。先设小规模预算验证接口，再另行确定完整训练预算。

## 7. 后续对照与通过条件

建议最小对照矩阵：

| 对照 | 输入帧数 | 用途 |
|---|---:|---|
| MATLAB mean / Taylor | 10 | 输入质量与无需网络的少帧基线 |
| 正式 no_mean 逐样本优化 | 10 | 同帧数下已有方法，200 步 |
| 共享网络，无 AlgoRIM 教师 | 10 | 检查多样本训练与 90 帧统计约束的价值 |
| 共享网络，加 AlgoRIM 辅助 | 10 | 仅在上组稳定后验证教师是否带来净收益 |
| 100 帧传统结果 | 100 | 多帧参考，不冒充与 10 帧方法同等采集预算 |

优先判断：深度是否正确、相对 VAR 是否真的提高可分辨细节、毛刺是否减少、不同子集是否稳定，而不是只看训练 loss 下降。每个新模块只改变一个主要变量，记录完整配置、样本划分、随机种子、best/final、时间和显存。

真实数据无真值时只能评估统计自洽、子集稳定性和模型敏感性，不能据此宣称绝对深度准确。若新方案未通过横向和轴向联合验证，继续保留 E0 no_mean 为正式基线。

## 8. 源码与证据索引

以下路径均相对于本文件所在项目根目录：

| 路径 | 核对内容 |
|---|---|
| `models/variance_anchored_lfm_net.py` | 主网络、beta、输出映射、候选开关 |
| `models/encoder3d.py`、`models/blocks.py` | 编码器、归一化、卷积模块 |
| `models/set_encoder.py` | mean/std Set 聚合与 Z lifting |
| `models/set_transformer_encoder.py` | 可选 Transformer 聚合 |
| `models/gated_fusion.py`、`models/decoder3d.py` | 三尺度融合与残差解码 |
| `models/output_parameterizations.py` | 候选轴向/层内解耦 |
| `losses/self_supervised_losses.py` | Taylor H2、未中心化 log 方差项、TV |
| `physics/lfm_operator.py` | 相位相关前向物理 |
| `train_volume.py` | 实际数据读取、优化循环、checkpoint 选择 |
| `configs/depth50_n100_no_mean_loss.yaml` | 正式默认配置；当前文件为 400 步 |
| `utils/visualization.py` | gate 平均图含义 |
| `docs/branch_ablation_50um.md` | 历史 400 步分支消融证据 |
| `docs/resolution_evaluation.md` | FTC 中心校正、旧结果失效说明 |
| `matlab_code/pilot_dataset/pilot_dataset_config.m` | 冻结数据生成协议与光学参数 |
| `matlab_code/pilot_dataset/pilot_build_subset.m` | 10/90 统计与 3 次 RL 的实际调用 |
| `matlab_code/pilot_dataset/pilot_reconstruct_frame.m` | 单帧初步重建 |
| `matlab_code/pilot_dataset/pilot_export_algorim.m` | 两种尺度版本的同深度 100 页栈 |

本文件中的“当前”仅指整理时源码与配置状态；候选存在不代表被默认启用，历史结果不代表本次重新运行验证。
