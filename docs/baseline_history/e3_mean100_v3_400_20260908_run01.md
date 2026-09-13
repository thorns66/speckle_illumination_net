# 当前正式基线：E3＋100% mean 结构梯度上限

用户已确认，将 E3＋100% 作为后续实验的新基线。默认参照是本轮 **第 400 步 final**；第 360 步 best 同时保留，单独比较。

- 可机器读取的模型、配置、数据指纹和 SHA256：[CURRENT_BASELINE.json](CURRENT_BASELINE.json)。
- 来源实验：[完整报告](outputs/v3_mean050_mean100_400_20260908_run01/REPORT_ZH.md)。
- 默认模型：[checkpoint_last.pt](outputs/v3_mean050_mean100_400_20260908_run01/e3_mean100/checkpoint_last.pt)，第 400 步。
- 补充模型：[checkpoint_best.pt](outputs/v3_mean050_mean100_400_20260908_run01/e3_mean100/checkpoint_best.pt)，第 360 步。
- 固定配置：[e3_mean100.yaml](outputs/v3_mean050_mean100_400_20260908_run01/e3_mean100.yaml)。
- 历史实现快照：`outputs/v3_mean050_mean100_400_20260908_run01/source_snapshot`。

## 方法定义

Taylor＋Mean＋Set 三分支及 Gate，十帧输入及对应 MATLAB RL3 初步重建；Taylor 读取已保存的 sqrt，只开方一次。十帧均值解析确定整体亮度，普通 mean 项只调整亮度参数，归一化方差项负责结构，同时加入逐样本 q 处最多与方差梯度等大的归一化 mean 结构梯度。系数上限为 1，前 50 步渐增；100% 不是固定的 loss 权重，也不要求实际梯度占比达到 100%。

保留 TV、原 PSF 和照明设置。光学参数尚待用户核实，沿用本次配置，不根据基线命名改动 NA。E3 行为依赖 `tools.mean_budget_experiment` 适配器；不能仅将 YAML 交给普通 no_mean 入口就认为复现了 E3。

## 后续对比约定

- “新基线”指此处的 E3＋100%；原 baseline、E3 和 E3＋5% 保留历史方法名，旧结果不改名、不覆盖。
- 同预算比较优先对照第 400 步 final；best 作为补充单列。
- 从头训练的消融继续使用原共同随机初始化；若从新基线权重继续训练，明确标记为微调。
- 正式数据划分维持 V3：P01–P11 训练，V01–V03 验证，T02–T04 测试，110/30/30 子集。
- 本次仅登记新基线，不启动训练，也不改动冻结实验入口或已有预测。

## 已知边界

当前选择基于一个训练种子。100% 组 final 的测试对象等权形状误差为 0.8169，轴向 W1 为 5.22 µm，背景占比约 0.38%。T04 的 20–40 µm 双层正式分开率仍为 0；旧点目标局部轴向 W1 仍逊于 E3。新基线的身份不代表这些问题已经解决。
