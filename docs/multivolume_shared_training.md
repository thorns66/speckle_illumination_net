# 多样本共享网络：训练与推理手册

> 当前正式基线已于2026-09-13由用户指定为 **Mean100 final800：Mean-RL3基础＋E3＋mean≤100%，保留Set，P01–P11＋真实45/55，不含P12**。以 [CURRENT_BASELINE.md](../CURRENT_BASELINE.md) 和 [机器清单](../CURRENT_BASELINE.json) 为准。下文配置、200步运行、空闲卡限制等属于历史实验协议，不自动适用于新基线。

## 历史基线选择（2026-09-07，已被2026-09-13登记替代）

用户在查看归一化对比图及单层效果后决定：继续保留 **sqrt VAR 特征 + Mean + Set
branch + Gate** 作为基线，不将无 Set 消融升级为主方案。

- 基线配置：`configs/multivolume_n10_no_mean.yaml`，`var_feature_representation=sqrt`、
  `use_set_branch=true`。`no_mean` 仅指 Mean 损失关闭，不表示移除 Mean 分支。
- 基线运行：`outputs/multivolume_n10_no_mean_run01`；训练 200 步，物理最佳 step160，
  权重为该目录中的 `checkpoint_best.pt`。默认训练入口仍使用这一配置。
- 无 Set 运行 `outputs/multivolume_n10_no_set_20260907_run01` 及其配置、权重、图和
  指标作为消融资料保留，不覆盖或替代基线。
- 无 Set 的 W1 和对齐 NRMSE 改善仍是有效的定量观察；但用户视觉验收认为分辨率
  没有提升、单层结果更脏，未达到替换基线的要求。该视觉反馈尚未单独量化为分辨率
  或伪影指标，不能改写为已通过定量检验的结论。
- W1 汇总横向像素后比较轴向质量分布；整体 NRMSE 也不直接等于分辨率或单层纯净度。
  后续选型不能仅凭这两项主指标胜出就自动替换基线。除非用户另行指定，新实验仍以
  上述有 Set 的 sqrt 方案为对照。

## 输出命名约定（2026-09-07 起）

所有新实验根目录使用 `outputs/<实验名>_<YYYYMMDD>_runNN`；日期为实际启动
当天的北京时间，同日重复运行递增编号，禁止覆盖。根目录中的权重、指标和报告
保留固定文件名；独立存放的汇总文件也加日期。历史结果不改名，本文后面的旧路径
仅代表历史运行。新启动命令必须采用此命名约定。

无 Set 消融使用 `configs/multivolume_n10_no_set.yaml`，只关闭 Set 及其专属门控
修正通路，VAR/Mean、sqrt 锚点、损失、初始化种子、全局 batch 和训练步数不变。
模块仍按原顺序构造以保持公共初始化，旁路参数在 Adam/DDP 创建前冻结。
`utils.experiment_paths.next_experiment_path` 生成北京时间日期及未使用编号。
使用 `python -m tools.run_no_set_ablation` 可自动核对基线、等待空闲 GPU、生成日期
目录并运行；完成后用 `python -m tools.report_no_set_ablation --run <新实验目录>`
生成比较。只使用没有其他计算进程的空闲卡，不添加 `--allow-busy-gpus`。

主比较按各自验证物理损失最优 checkpoint，补充固定第 200 步验证及完整轨迹。
先对子集平均，再对对象等权平均。单种子结果只表示本实验下整个 Set 通路的贡献，
不能证明统计显著，也不能区分额外输入信息与额外模型容量的作用。

可视化默认提供全体积最大值归一化的清晰版（各方法独立归一化至 0–1，统一显示
gamma=0.5）及线性版（gamma=1），原始强度图保留用于审计。不逐层或逐投影归一化，
不阈值裁背景；gamma 只作用于颜色，不改变深度曲线、训练输入、原始重建和评价指标。
归一化图用于比较结构，不能比较方法间的绝对亮度。对已有结果只重绘而不重新训练：
`python -m tools.plot_no_set_normalized --run <实验目录>`，输出自动带日期和编号，
覆盖全部 30 个固定测试子集，并核对绘图前后源数据、权重及指标文件哈希一致。

## 1. 已冻结的实验定义

正式配置为 `configs/multivolume_n10_no_mean.yaml`，其定义不是单样本 DIP：

- 一个训练条目是“一个完整物体 + 一个固定 10 帧子集”。
- 输入只来自这 10 帧：MATLAB 三次 RL 得到的 `physics_taylor_sqrt_float`
  （VAR 分支）、`physics_mean_raw`（Mean 分支），以及 10 张
  `sensor_pre_detector` 减去自身均值后的无序帧集（Set 分支）。
- 同一物体剩余 90 帧的无偏样本方差是训练约束；90 帧不会进入网络输入。
- 损失保持正式 no_mean 基线：`lambda_mean=0`、绝对 log 方差损失、
  `H²(g²)`、弱 3D TV。Mean 分支仍是输入特征，但均值误差不参与反传或选模。
- 网络保持 VAR + Mean + 原 mean/std SetBranch + Gate，以及 V1
  `legacy_anchor_positive` 输出参数化。没有启用失败的 V2/V3、Set Transformer、
  detail head 或轴向先验。
- `beta0` 对每个 10 帧输入由其均值解析计算；网络只学习一个所有样本共享的
  有界相对修正 `raw_beta`，没有逐层 `beta_z`。
- 全局 batch 固定为 8。无论选择 1–6 张 GPU，每一步都使用相同的 8 个训练条目。
- 验证/测试按整个物体隔离：train=P01–P06/P08/P10，validation=P09/V01/V02，
  test=P07/T01/T02。每个物体的 10 个子集先求均值，再在物体间求均值。
- GT 只计算报告指标，绝不进入输入、loss 或 checkpoint 选择。

## 2. 一键启动正式训练

先进入工程和环境：

```bash
cd /workspace/xyx/speckle_illumination_net
conda activate speckle_net
```

推荐使用全部六张空闲 A40：

```bash
python -u train_dataset.py \
  --config configs/multivolume_n10_no_mean.yaml \
  --gpus all \
  --output-dir outputs/multivolume_n10_no_mean_run01
```

也可以按 `nvidia-smi` 的物理编号任意选择。顺序会被原样记录，例如：

```bash
# 一张卡
python -u train_dataset.py --config configs/multivolume_n10_no_mean.yaml \
  --gpus 4 --output-dir outputs/multivolume_n10_no_mean_gpu4

# 三张指定卡
python -u train_dataset.py --config configs/multivolume_n10_no_mean.yaml \
  --gpus 0,2,5 --output-dir outputs/multivolume_n10_no_mean_gpu025
```

启动器会先打印并保存物理编号、UUID 和启动时负载。默认拒绝显存超过
1024 MiB 或利用率超过 10% 的卡，避免误抢其他任务。确实要使用忙卡时才添加
`--allow-busy-gpus`。不要再手工设置 `CUDA_VISIBLE_DEVICES`；启动器使用 UUID
完成物理卡到本地 rank 的映射。

此服务器的六张 A40 分属两个 PCIe/NUMA 岛且没有 NVLink。启动器已采用实测
通过的 `NCCL_P2P_DISABLE=1`、`NCCL_IB_DISABLE=1`，避免 NCCL 初始化卡死。
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 在当前驱动可能报告“不支持”；
这是警告，不影响训练。实测 phase chunk 32 的峰值为 32.83 GiB，低于 42 GiB
门槛；若以后模型变化导致 OOM，预检会自动降为 16。

首次运行会创建两种可复用缓存：

- `data/.cache/matlab_multivolume_n10/`：已严格校验的 10/90 数据条目，约 1.6 GiB。
- `data/.cache/psf/selected_H_*.npy`：10 个深度的未压缩 PSF，约 3.5 GiB。

PSF 缓存由一个进程原子生成，随后所有 DDP 进程通过操作系统页缓存共享读取，
避免六进程重复解压 HDF5。

## 3. 断点续训

使用上一次相同数量的 GPU；物理编号可以换，但数量必须相同：

```bash
python -u train_dataset.py \
  --config configs/multivolume_n10_no_mean.yaml \
  --gpus all \
  --resume outputs/multivolume_n10_no_mean_run01/checkpoint_last.pt
```

若希望把原计划延长到 300 步：

```bash
python -u train_dataset.py \
  --config configs/multivolume_n10_no_mean.yaml \
  --gpus all \
  --resume outputs/multivolume_n10_no_mean_run01/checkpoint_last.pt \
  --max-steps 300
```

续训会恢复模型、Adam、固定全局 batch 调度器、Python/NumPy/PyTorch/CUDA RNG，
并保留旧 CSV。除 `max_steps` 和实际输出目录外，配置发生任何变化都会直接拒绝，
防止把两个实验混到同一个目录。正式配置每 20 步验证并保存一次，因此异常中断
最多回退到最近的 20 步 checkpoint。

## 4. 输出与进度

主要文件：

- `checkpoint_best.pt`：按验证物体平均的 `weighted_var + weighted_tv` 选择。
- `checkpoint_last.pt`：最近一次可精确续训的状态。
- `training_metrics.csv`、`validation_metrics.csv`、`test_metrics.csv`。
- `validation/step_*/.../reconstruction.{npy,tif}` 和对应 test 输出。
- `training_complete.json`：完成步数、best step 和最终测试汇总。
- `run_contract.json`、`gpu_selection.json`、`config_used.yaml`：完整追溯记录。
- `tensorboard/`：训练、验证和最终测试标量。

另开终端查看：

```bash
watch -n 1 nvidia-smi
tensorboard --logdir outputs/multivolume_n10_no_mean_run01/tensorboard --port 6006
```

## 5. 只用 10 帧进行独立推理

```bash
python -u infer_dataset.py \
  --checkpoint outputs/multivolume_n10_no_mean_run01/checkpoint_best.pt \
  --sample-dir data/matlab_cells_pilot_v2_r04/P07 \
  --subset 1 \
  --gpu 4 \
  --output-dir outputs/inference_P07_subset01
```

该入口只打开 subset 中的输入索引、10 帧 MATLAB 初重建和对应 10 张传感器帧；
不会读取 holdout mean/variance 或 ground truth。输出包括重建 TIFF/NPY、输入
F_var/G_mean NPY 和 `inference_contract.json`。

## 6. 已完成验证（2026-09-06）

- MATLAB 深检和独立 Python SHA-256 审计：14 个物体、5,754 个不可变文件，0 错误。
- 全仓库 unittest：137 项通过；1 项需要外部 MATLAB 导出夹具，按设计跳过。
- 单卡 260×260 完整前后向：峰值 32.83 GiB，训练/验证/测试/checkpoint 全通过。
- 六卡 DDP、global batch=8：2/2/1/1/1/1 条目分配完成，无填充物理计算。
- 同一全局 batch 的单卡/六卡模型更新最大绝对差 `1.1641532e-10`。
- 独立推理与训练器内部同 checkpoint 输出逐元素相同，最大绝对差 0；TIFF 回读相同。
