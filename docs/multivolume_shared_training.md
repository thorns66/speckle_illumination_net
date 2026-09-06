# 多样本共享网络：训练与推理手册

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
