# SeReNet / VCD-Net 散斑光场三维重建基线

这是独立于现有训练工程的比较目录。两个网络均从固定的10帧物理强度图像取均值，输出深度10–100 μm、步长10 μm的三维体。正式训练默认800 epochs；本目录的验收权重只用于检查程序，不是论文比较结果。

## 直接开始训练

在项目根目录执行，按当前GPU空闲情况修改 `--gpu`。两个命令分别使用各自的Conda前缀环境，不需要手动激活环境。

```bash
bash comparison_methods/scripts/train_serenet.sh --gpu 3
bash comparison_methods/scripts/train_vcdnet.sh --gpu 3
```

建议依次运行。默认输出分别为 `comparison_methods/outputs/serenet_seed20260901` 和 `comparison_methods/outputs/vcdnet_seed20260901`。已有检查点的输出目录会拒绝被新训练覆盖。另开实验可传入 `--seed 20260902 --output /绝对路径/新实验目录`。

续训示例（需要保持原训练seed、学习率和fixed-sample设置）：

```bash
bash comparison_methods/scripts/train_serenet.sh --gpu 3 \
  --resume comparison_methods/outputs/serenet_seed20260901/checkpoint_last.pt
```

`--max-steps 20` 表示本次进程再执行20次更新，用于短检查；省略时完成配置中的全部epochs。正式训练不要传 `--validation-limit` 或 `--fixed-sample`。默认Adam、学习率1e-4、batch size 1、FP32，每10 epochs进行完整验证并保存检查点。保存模型、优化器、随机状态、当前epoch中的样本顺序和游标。默认验证分别按物理投影MSE和三维MSE选择最佳模型，不使用测试集。

## 数据与监督

| 项目 | SeReNet | VCD-Net |
|---|---|---|
| 输入 | 固定10帧均值 | 同一组10帧均值 |
| 训练目标 | 自身均值的物理前向投影一致性＋非负惩罚 | 仿真三维真值MSE |
| 三维真值参与训练 | 否 | 是 |
| 其余90帧参与训练 | 否 | 否 |
| 输入通道 | 49×49＝2401个角度 | 49×49＝2401个角度 |
| 输出 | `[B,1,10,H,W]` | `[B,1,10,H,W]` |

冻结划分为：P01–P11训练（110个子集）、V01–V03验证（30个）、T02–T04测试（30个）。所有对象的帧和衍生数据属于同一划分；P12排除。真实45/55视野仅供这两个基线迁移推理。如果你的V5模型已经用45/55训练，不能把这些视野称为V5的未见测试数据。

数据来源为项目中已有的 `outputs/v3_mean_anchor_e3_mean100_400_20260909_run01/dataset_v3_frozen_view`。均值从 `sensor_pre_detector` 浮点数据重算并与已存均值逐值核对，不使用展示TIFF。只在VCD的独立数据路径读取训练GT，没有修改原项目的数据读取保护。

`manifests/data.json` 记录对象、帧索引、缓存哈希、PSF哈希与尺度。输入尺度为训练集所有10帧均值的最大值。VCD目标尺度为训练GT最大值的1.05倍，通过 `(tanh_output+1)/2` 映射并恢复物理尺度。SeReNet的输出尺度仅用训练输入尺度和PSF对全1体的平均响应计算，完全不依赖GT。验证、测试、真实推理均不拟合新尺度、不根据测试GT调亮度。

缓存已准备。重建缓存执行：

```bash
bash comparison_methods/scripts/prepare_data.sh
```

输入为260×260时，只在右、下补零至294×294，按行优先角度顺序拆为2401×6×6，保留有效掩码。网络输出裁回原区域后才计算物理投影和损失，不把补零区当成观测。真实1029×1421输入拆为2401×21×29，保持全视野和像素尺寸。

## 实现来源与适配边界

- SeReNet：https://github.com/kimchange/SeReNet ，提交及文件哈希见 `manifests/sources.json`。保留深度分解、PSF重聚焦、三维融合和投影自监督；用本系统2401角度、10深度和49倍空间采样替换示例参数。
- VCD原作者源码：https://github.com/feilab-hust/VCD-Net ，下载保存在 `upstream/VCD-Net`，用于对照；实际训练采用SeReNet论文比较中的PyTorch移植版，输入2401通道、输出10层，`channels_interp=128`。不是TensorFlow原版逐算子复现。
- 原始网络文件不被修改；适配器加载其类定义，仅移除注册装饰器与无用工具导入，避免与现有工程同名模块冲突。所有有意义的网络层仍来自固定上游文件。
- SeReNet当前采用归一化投影MSE＋10倍平方负值惩罚，没有启用原论文NLL-MPG、LPIPS、TW-Net、preDAO或F-SeReNet。原因是当前数据为探测器噪声加入前的仿真均值；不得把此配置写成完整原实验复现。
- 投影采用已有周期移变稀疏算子和精确伴随，读取原缓存，不重新解释PSF为普通空间不变卷积核。推理无需加载完整物理算子。
- 重聚焦使用相位(24,24)处点源PSF各角度的质心偏移，统一对齐至微透镜中心；不沿用其他显微镜的几何常数。

论文建议名称：**SeReNet（本系统自监督适配）**、**VCD-Net（SeReNet作者PyTorch实现，本系统监督适配）**。相同10帧采集预算不等于信息利用完全相同：均值输入丢失帧间变化，而你的网络使用该变化。两种基线的监督条件也不同，需要同时披露。有限散斑帧的平均照明并非严格均匀，是均值光场基线在本任务中的模型近似。

许可证原文保留在上游目录；SeReNet检出的源码中没有独立LICENSE文件，来源清单如实记录，不为其自行指定许可证。

## 推理与评估

一个仿真测试样本：

```bash
bash comparison_methods/scripts/infer.sh serenet --gpu 3 \
  --checkpoint comparison_methods/outputs/serenet_seed20260901/checkpoint_best.pt \
  --object T02 --subset 1 --output comparison_methods/outputs/serenet_test
```

完整30个仿真测试子集：省略 `--object` 和 `--subset`。真实视野用 `--field 45 --subset 1`；VCD将方法名和检查点目录替换为 `vcdnet`。真实图像全视野推理，不自动缩图。

```bash
bash comparison_methods/scripts/evaluate.sh serenet \
  --predictions comparison_methods/outputs/serenet_test
```

输出float32 ZYX TIFF、深度与像素尺寸元数据、逐样本计时和显存。非负导出使用零截断，不做逐图拉伸。计时拆分为输入读取／传输、完整模型前向、含导出的总时间；模型前向包括角度拆分和SeReNet重聚焦。首次调用可能包含CUDA初始化开销，验收计时不是稳态速度基准。

评估采用固定训练GT尺度，报告归一化MSE、MAE、PSNR和逐层二维SSIM的平均值（明确不是三维SSIM），先按对象汇总再取对象平均。仅部分测试图像时 `test_complete=false`，不伪装为完整测试。真实数据没有GT，不计算PSNR／SSIM。

## 独立环境与验收

两个环境位于 `.envs/serenet`、`.envs/vcdnet`，均为独立文件副本，Python 3.11、PyTorch 2.5.1+cu121、torchvision 0.20.1+cu121。精确pip及Conda清单保存在 `environments/`；`existing_environment_before/after.txt` 用于证明原环境依赖未变。启动器禁用用户级Python包并清除外部PYTHONPATH。

```bash
bash comparison_methods/scripts/setup_envs.sh
bash comparison_methods/scripts/preflight.sh serenet 3
bash comparison_methods/scripts/preflight.sh vcdnet 3
```

可附加 `--cpu` 进行CPU预检，但CPU通过不代表GPU训练就绪。程序只使用指定GPU，不终止现有进程；显存不足时选其他设备或稍后运行，不会悄悄缩小数据。

GPU验收包括光场拆分还原、点源投影／伴随验证、每个网络20步训练＋2步续训、100步固定训练样本过拟合，以及仿真和真实全视野推理。短训练只验证链路、梯度和收敛方向，正式性能仍需完整训练与独立测试。最终实测状态见 `ACCEPTANCE_ZH.md` 和 `outputs/acceptance.json`。

计算可复现性：固定种子、样本顺序并开启cuDNN确定性卷积。PyTorch 2.5.1的CUDA双线性／三线性插值反向仍存在非确定性；保留原网络算子，因此不承诺独立重复训练逐位一致。验收会在续训更新之前精确核对模型、优化器和全部随机状态已正确恢复。
