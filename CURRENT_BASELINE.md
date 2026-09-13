# 当前正式基线：Mean100 · Mean主分支 · final800

登记日期：2026-09-13（北京时间）。用户明确要求“把当前mean100作为新的baseline基线并记录”。
唯一当前ID：`mean100_v5_mean_anchor_mixed_real_no_p12_800_20260913`。

## 默认模型：final800，不是best740

- 默认权重：[checkpoint_step_000800.pt](outputs/v5_sim_real_no_p12_mean_anchor_e3_mean100_800_20260911_run01/mean_anchor_e3_mean100/checkpoint_step_000800.pt)。
- SHA256：`ae61ff74cb40cc34dfeac0c221b13cdbfc6e4e9b84514e5282e3ea6ed7dafa03`；与同目录 `checkpoint_last.pt` 完全一致。
- [冻结有效配置](outputs/v5_sim_real_no_p12_mean_anchor_e3_mean100_800_20260911_run01/mean_anchor_e3_mean100/config_used.yaml)。
- [best740](outputs/v5_sim_real_no_p12_mean_anchor_e3_mean100_800_20260911_run01/mean_anchor_e3_mean100/checkpoint_best.pt) 仅作补充，不自动代替用户指定的final800。
- 完整路径、模型/数据/PSF指纹及复现入口见 [CURRENT_BASELINE.json](CURRENT_BASELINE.json)。

## 方法与训练协议

Mean100指 **Mean-RL3重建基础＋Taylor-sqrt／Mean／Set三分支＋Gate＋E3＋mean≤100%结构梯度预算**。
100不是输入帧数、迭代次数或固定loss权重。输入仍是10帧及对应RL3，Taylor编码器仍用反卷积后sqrt；Set未删除。

重建：`u=P(beta_mean*g_mean, residual)`，`q=u/sum(u)`，`g_hat=a*q`。
beta从所选Mean基础的全幅投影与输入10帧均值解析计算，再作有界修正；E3亮度也由输入10帧决定。
VAR、Mean、Set提供残差特征，最终基础是Mean，不是Taylor。90帧目标只用于损失，不进入推理。

- 仿真训练P01–P11（110子集）；真实45、55各10子集；P12排除。
- 每步6仿真＋45、55各1真实子集，global batch8；真实全幅1029×1421，不缩放、不切块。
- 验证V01–V03、测试T02–T04，各30子集。45/55编号不代表深度，二者均无GT且均参与训练。
- 同一共同初始化，seed20260901；800步，Adam、FP32，关闭AMP/TF32。
- 第1–200步网络LR1e-3、beta/gain LR1e-4；之后1e-4、1e-5。mean结构梯度预算100%，前50步渐增；TV1e-5、轴向系数0.5。
- 原PSF十层10–100 μm、间隔10 μm；零噪声参数沿用现有未标定设置，不因本次登记改变。

## 结果入口与边界

- [2026-09-13四方法整体分析](outputs/mean100_baseline_analysis_20260913_run01/analysis/REPORT_ZH.md)：Mean-RL3、Taylor-RL3-sqrt、Taylor100 final800、Mean100 final800；每方法94例，另含Mean best740补充及45/55训练视场诊断。28本分组图册、十层图、局部分辨、稳定性和完整训练曲线均已验收。
- 本次测试对象等权结果：Mean100对齐NRMSE 0.78386、轴向W1 4.82650 μm，优于Taylor100的0.82334、5.68254 μm；但背景与跨子集稳定性存在退步，T04局部双层分辨仍失败。结论为混合收益，不因基线登记而宣称全面胜出。
- [菠菜根完整六方法报告](outputs/spinach_v5_800_detailed_20260913_run02/REPORT_ZH.md)，40个RL案例＋80个网络案例，45/55各10子集。
- [按用户要求拆分的对比图册](outputs/spinach_grouped_comparison_20260913_run01/README_ZH.md)。
- 训练目录内自动生成的 `test_metrics.csv` 是 **best740** 的仿真测试，不能标为final800测试。
- 晋升依据是用户明确选择；不宣称单种子结果统计显著，不宣称真实深度或分辨率已被GT证实。真实训练约束一致性不等于独立测试误差。
- 与旧400步模型比较同时改变了数据和训练预算，不能全部归因于真实数据。

## 后续任务如何使用

所有新实验、推理和写作任务先重新读取本文件及JSON；默认比较使用final800。若指定旧实验，保留其冻结定义，不擅自切换权重。
现有800步训练入口为 `tools.queue_v5_taylor_then_mean_800`（Mean worker），方法适配器为 `tools.v5_mixed_real_anchor_experiment`。
该适配器的历史通用校验仍限定600步，不能把YAML直接交给普通入口就认为能复现800步；应核对800步包装器和冻结源码。
已核验真实推理路径为 `tools.spinach_v5_800_detailed`，必须保留全幅beta、E3归一化/亮度，不可只调用裸模型输出作为最终结果。

旧Taylor400正式登记已[归档](docs/baseline_history/e3_mean100_v3_400_20260908_run01.md)，旧结果和历史Git标签不改动。
旧 `tools/run_mean100_extend_noset.py` 固定校验旧基线ID；它对新ID报错是预期保护，不应删掉检查后误用。

## 跨Codex窗口同步

根目录 [AGENTS.md](AGENTS.md) 要求每个相关任务重新读取当前注册。共享同一工作树的窗口共享这些文件；已运行的会话未必立即刷新上下文。
可在其他窗口发送：**“重新读取项目AGENTS.md和CURRENT_BASELINE.md/json，按其中Mean100 final800更新当前基线；不改变正在执行的冻结实验。”**
不同机器、克隆或worktree需同步这些文件后再读取。本次未提交或推送Git，也没有向其他会话注入消息。

变更审计见 [2026-09-13登记记录](docs/baseline_history/promotion_mean100_20260913.md)。
