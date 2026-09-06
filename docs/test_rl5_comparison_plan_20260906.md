# 测试集 Mean / Taylor / 网络对比方案

日期：2026-09-06。固定对象 P07、T01、T02；固定网络为
`outputs/multivolume_n10_no_mean_run01/checkpoint_best.pt`（step 160）。
现已完成 100 帧 Mean/Taylor RL5 的 6 个结果，以及 10 帧固定子集 RL5 的 60 个结果。
完整评价共 156 条逐体记录（另含 30 个网络结果及 60 个 RL3 控制），见
[完整中文报告](../outputs/test_rl5_baselines_20260906/evaluation_complete/report_zh.md)。
下文第 8 节保留早期 100 帧阶段结果，第 9 节记录最终交付。

## 1. 必须回答的三个问题

1. **相同 10 帧输入，网络能否超过传统方法？** 主比较为每个固定子集的
   Network-10、Mean-10-RL5、Taylor-10-RL5，逐子集配对。
2. **网络是否只相当于多做两次反卷积？** 保留现有 Mean-10-RL3 和
   Taylor-10-RL3，同时比较 RL3→RL5 与 RL3→Network 的变化。
3. **只用 10 帧，离 100 帧传统方法还有多大差距？** 与 Mean-100-RL5、
   Taylor-100-RL5 比较；这是采集帧数参考，不能称为公平同帧数比较或理论上限。

## 2. 对比矩阵

| 方法 | 每次使用的测试帧数 | 迭代 | 数量 | 作用 |
|---|---:|---:|---:|---|
| Mean-10-RL3 | 对应子集的 10 帧 | 3 | 30 个已有体 | 网络 Mean 输入基线 |
| Taylor-10-RL3 | 同一 10 帧 | 3 | 30 个已有体 | 网络 VAR 输入基线 |
| Mean-10-RL5 | 同一 10 帧 | 5 | 30 个已补体 | 同帧数主对照 |
| Taylor-10-RL5 | 同一 10 帧 | 5 | 30 个已补体 | 同帧数主对照 |
| Network-10 | 同一 10 帧；输入体仍为 RL3 | 已训练 step 160，测试仅前向 | 30 个已有体 | 待评价方法 |
| Mean-100-RL5 | 该物体全部 100 帧 | 5 | 3 个本轮体 | 多帧参考 |
| Taylor-100-RL5 | 该物体全部 100 帧 | 5 | 3 个本轮体 | 多帧参考 |

本轮共完成 66 个新增体：6 个全 100 帧 RL5，以及 60 个 10 帧子集 RL5。
不需要为这个对比把每张单帧的独立 RL 重建重跑一遍，也不需要先修改训练集或重新训练。

## 3. 重建定义和复现约束

- Mean：在传感器域对 N 帧 `sensor_pre_detector` 求均值，用 H/Ht 重建 5 次。
- Taylor：先计算同一 N 帧的无偏样本方差（N−1 分母），用 H.^2/Ht.^2
  重建 5 次，最后对非负 raw 输出开方一次。保存开方前后数据。
- 直接复用 `pilot_reconstruct_volume` 及历史 `deconvRL`，初始值为 Ht(y)。
  当前项目所谓 RL 的具体更新是 `X *= Hty / Ht(HX)`；报告需标明沿用这一历史实现，
  不把它默认为另一种标准 RL 更新。后续若要比较其他 RL 实现，应独立命名和验证。
- 100 帧先求统计量再反卷积；不使用“逐帧反卷积后求均值/方差”替代。
- 固定原始 10 套输入索引、不重抽样；10 帧基线不使用其余 90 帧。
- 固定 PSF、10–100 µm 的十层网格、260×260 图像和数值精度。
- 保留 physics 浮点尺度：double 累加统计量、single 求解器，无逐帧/逐层 max 归一化。
- 若旧求解器产生超出容差的负值，沿用已有非负数值回退，记录是否触发及耗时。
- 输出在独立 `outputs/test_rl5_baselines_20260906`，不覆盖原始 RL3 subset MAT、
  数据缓存、训练 checkpoint 或原数据清单。运行前后核对来源文件 SHA-256。
- 网络继续使用 RL3 输入。直接把其输入替换成 RL5 会改变输入分布，是另一项实验。
- 本实验无额外相机噪声，结论适用于这批仿真；对照方法没有跨物体训练，网络有训练成本。

## 4. 评价口径

主指标使用相同十层粗网格真值 `prepared.mat:ground_truth`。
100 层细网格只用于几何展示，不将其当作网络具有 1 µm 轴向采样能力的证据。

1. **三维结构误差**：全体积 scale-aligned NRMSE。每个输出只允许一个全局正增益，
   不进行逐层缩放、不做几何配准。另列 raw NRMSE/增益值，区分幅度和结构。
   真值拟合的增益只用于评价，不写回重建，也不用于推理、训练或 checkpoint 选择。
2. **轴向分布**：层质量曲线、轴向 W1（µm）、层质量 L1、真值支持扩展 ±10 µm
   之外的质量比例。多层 P07 的支持外比例可能很小，仍需结合 W1/L1。
3. **单层目标**：T01（60 µm）、T02（90 µm）列出峰值层、质心、真实层质量、
   真值 ±10 µm 质量。深度指标不能被 XY MIP 相似性代替。
4. **横向结构**：同时展示原尺度 MIP SSIM 和同一全局增益对齐后的 MIP SSIM。
   SSIM 在稀疏图中容易受大面积背景影响，不能作为唯一质量或分辨率指标。
5. **后续局部分析**：T01 预先固定点/短丝位置测横向与轴向 FWHM、成对结构分离；
   P07/T02 预先固定细丝剖面、断口与交叉位置，观察展宽、粘连、假连接和漏检。
   ROI 由真值几何一次固定，对所有方法相同，不为每个结果挑最好看的位置。
6. **代价**：输入帧数、MATLAB 初重建时间、网络纯前向时间和端到端时间分开报告；
   PSF 加载/暖启动另列；200 步训练总耗时不等于单个测试物体推理耗时。

## 5. 汇总和可视化

- 同帧数 10 帧方法逐对象、逐 subset 配对计算差值；每个对象给 10 个结果的均值±SD，
  再在三个对象上等权宏平均。N=3 是独立物体数，不能把 30 套子集称为 30 个独立样本。
- 100 帧方法每个对象只有一份结果；不复制成 10 个结果增加样本量，
  与网络 10 个子集的分布并列展示。
- 主图固定 subset_01 展示一次 10 帧推理，不平均 10 份网络输出后当成 N=10 结果。
  全部 10 个子集图作为补充；原始重建体用于量化，图像显示归一化只作标注。
- 每个对象提供 GT、Mean10-RL5、Taylor10-RL5、Network10、Mean100-RL5、
  Taylor100-RL5 的 XY MIP；再提供 XZ/YZ MIP、真实层与统一剖面、完整 z 质量曲线。
- 同对象、同评价方式固定显示增益和色标；不逐层增强，不用单独挑选的预测峰层掩盖错层。
- 输出逐样本 CSV、对象汇总 CSV、来源清单、float32 MAT/TIFF/NPY 和可导出的 PNG。

## 6. 判定与边界

- 若 Network10 超过 RL3，但没有超过 RL5，只能说改善了现有三次迭代初值，
  还没有证明优于五次迭代的同帧数传统方法。
- 若 Network10 在结构/轴向误差上超过 Mean10-RL5 和 Taylor10-RL5，且局部细节
  没有增加假结构，可报告这三个测试对象上的同帧数增益，分别列出失败对象。
- 若 Network10 接近或超过 100 帧参考，可报告少帧优势；100 帧本身不是可靠真值。
- 5 次只是本轮冻结的迭代预算。若需要声称超过充分调优的传统方法，下一步应在
  validation=P09/V01/V02 上比较 3/5/10/20 等迭代数，按预先约定规则定下来再评估。
  不在 P07/T01/T02 上选最有利的迭代数、网络 checkpoint 或后处理。
- 本轮已经查看过测试表现；据此继续改模型属于探索。正式泛化结论需新增未查看过的
  测试物体/独立采集，而不能反复在这三个对象上调参后称为独立验证。

## 7. 执行入口

在仓库根目录，Python 入口通过 GPU UUID 选择卡，默认拒绝忙卡，始终拒绝覆盖已有结果。
本次用户允许显存充足时共用、要求空闲卡优先；恢复批次使用 `--max-workers 5
--subset-chunk-size 2 --allow-busy-gpus`，每批重新检查空闲卡。1/2 号卡空闲后已优先使用。
下列命令是首次执行模板；当前已完成的输出目录会被保护而拒绝重跑。

```bash
python3 tools/run_test_rl5_baselines.py \
  --repo /workspace/xyx/speckle_illumination_net \
  --output /workspace/xyx/speckle_illumination_net/outputs/test_rl5_baselines_20260906 \
  --scope full100

# 下一批：60 个固定 10 帧子集 RL5 对照；输出与 full100 分开。
python3 tools/run_test_rl5_baselines.py \
  --repo /workspace/xyx/speckle_illumination_net \
  --output /workspace/xyx/speckle_illumination_net/outputs/test_rl5_baselines_20260906 \
  --scope subsets10

/workspace/xyx/.conda/envs/speckle_net/bin/python tools/evaluate_test_rl5_baselines.py \
  --repo /workspace/xyx/speckle_illumination_net \
  --output /workspace/xyx/speckle_illumination_net/outputs/test_rl5_baselines_20260906
```

## 8. 早期 100 帧阶段结果（历史记录）

6/6 MATLAB 任务完成，均无需非负回退；340 个来源文件运行前后哈希一致。
Python 独立复算 100 帧均值/样本方差，与 MAT 保存值一致；float TIFF 与 MAT 重建逐元素一致；
Taylor 开方一次的语义通过验证；现有网络指标从重建 NPY 复算与训练报告一致。

下表为三个对象等权平均；网络先在每个对象的 10 个固定子集上平均。

| 方法 | 3D 对齐 NRMSE↓ | 轴向 W1 µm↓ | 支持外质量↓ |
|---|---:|---:|---:|
| Network-10 | 0.9071 | 15.173 | 29.45% |
| Mean-100-RL5 | 0.9365 | 22.193 | 44.46% |
| Taylor-100-RL5 | 0.9440 | 25.185 | 48.65% |

网络在这三项宏平均上更好，但 P07 的 3D NRMSE 为 0.9405，略差于
Mean100 的 0.9355 和 Taylor100 的 0.9345，不能声称所有对象全面胜出。
网络在 T01/T02 的轴向分布仍明显扩散。完成下一批同帧数 RL5 对照后再判断同帧数增益。

详细产物：`outputs/test_rl5_baselines_20260906/evaluation/summary.md`、
`metrics_per_item.csv`、`metrics_per_object.csv`、`metrics_macro.csv`、`verification.json`。
图像包括线性共用色标图、对所有方法使用相同 gamma=0.4 的显示增强图，以及轴向质量曲线。

## 9. 完整交付（2026-09-06）

60/60 子集 RL5 重建完成；30/30 两子集计算批次成功，约 39.8 分钟墙钟时间。
来源文件运行前后哈希一致，60 个新增子集结果均未触发非负回退。
完整复算通过固定输入索引、从原始帧重算统计量、Taylor 开方一次、TIFF/MAT 一致性与网络旧指标复现检查。
原始数据集、RL3 输入和训练 checkpoint 均保留；中断尝试的日志与空目录归档在输出内，不覆盖历史结果。

| 方法 | 3D 对齐 NRMSE↓ | 轴向 W1 µm↓ | 支持外质量↓ | 对齐 MIP SSIM↑ |
|---|---:|---:|---:|---:|
| Mean10-RL5 | 0.9389 | 22.139 | 44.36% | 0.8186 |
| Taylor10-RL5 | 0.9520 | 25.207 | 48.64% | 0.8024 |
| Network10 | 0.9071 | 15.173 | 29.45% | 0.8442 |
| Mean100-RL5 | 0.9365 | 22.193 | 44.46% | 0.8188 |
| Taylor100-RL5 | 0.9440 | 25.185 | 48.65% | 0.8010 |

网络相对同帧 Mean/Taylor RL5 的宏平均 NRMSE 下降 3.38%/4.72%，W1 下降 31.47%/39.81%。
但是 P07 全体积 NRMSE 不及 Mean10 和两个 100 帧参考；T01 的网络峰值层为 80 µm 而真值为 60 µm；
T02 断口固定 ROI 的网络 NRMSE 为 0.9819，高于 Mean10 的 0.9392 和 Taylor10 的 0.9524。
因此不能声称所有对象、局部结构或光学分辨率全面提升。

最终入口：`outputs/test_rl5_baselines_20260906/evaluation_complete/report_zh.md`。
目录含 156 条逐体指标、对象均值±SD、配对差值、全部 10 子集 XY 图、XZ/YZ、逐深度图、固定 ROI、
点剖面、训练曲线和来源核验，共 22 张补充科学图；原始重建为 float32 MAT/TIFF/NPY。

仅重建报告（不重跑 RL5）：

```bash
/workspace/xyx/.conda/envs/speckle_net/bin/python tools/evaluate_test_rl5_baselines.py \
  --repo /workspace/xyx/speckle_illumination_net \
  --output /workspace/xyx/speckle_illumination_net/outputs/test_rl5_baselines_20260906 \
  --report-dir /workspace/xyx/speckle_illumination_net/outputs/test_rl5_baselines_20260906/evaluation_complete \
  --require-subsets
/workspace/xyx/.conda/envs/speckle_net/bin/python tools/report_complete_rl5_comparison.py \
  --repo /workspace/xyx/speckle_illumination_net \
  --output /workspace/xyx/speckle_illumination_net/outputs/test_rl5_baselines_20260906 \
  --report-dir /workspace/xyx/speckle_illumination_net/outputs/test_rl5_baselines_20260906/evaluation_complete
```
