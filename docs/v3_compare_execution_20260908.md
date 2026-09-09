# V3 三组各 400 步实验

本轮已按共同随机初始化（20260901）、全局 batch 8、micro batch 1、完整 FP32 运行。实际空闲卡是 A40 1–5；GPU 0 被其他任务占用，未使用。三组顺序训练，结束后各用一张空闲卡并行推理。

入口：`tools/run_v3_compare.py`。独立结果根目录：

`outputs/v3_baseline_e3_mean005_400_20260908_run01`

```bash
/workspace/xyx/.conda/envs/speckle_net/bin/python tools/run_v3_compare.py --dry-run
/workspace/xyx/.conda/envs/speckle_net/bin/python tools/run_v3_compare.py --stage all --gpus auto --resume
/workspace/xyx/.conda/envs/speckle_net/bin/python tools/run_v3_compare.py --stage infer --experiment all --gpus auto --resume
/workspace/xyx/.conda/envs/speckle_net/bin/python tools/run_v3_compare.py --stage report --resume
```

已有完整结果时应使用 `--resume`。配置、数据指纹和训练源文件不匹配时禁止继续混用；不要修改 YAML 后在原目录续训。新的一轮须设独立输出目录，并重新生成预检及来源记录。

训练源代码与配置保存在 `source_snapshot/` 和 `preflight.json`；这次未修改历史模型、物理算子、数据生成文件或旧训练结果。`v3_compare_experiment.py` 只在其进程内给现有训练器安装实验适配器。

三组公共网络参数完全相同，B/C 的额外 gamma 初始值为零。前 200 步网络/标量学习率为 1e-3/1e-4，第 201 步起为 1e-4/1e-5，Adam 状态连续。每 20 步全验证集打分并写检查点；滚动 last、best 以及单独 step200 被保留。final 是 step400 的 checkpoint_last。

按数据集原生 V3 读取器使用 P01–P11、V01–V03、T02–T04，分别 110/30/30 个子集。旧诊断场景仅用作补充评价。GT 不参与训练或 best 选择。

`v3_compare_checks.py` 完成训练前检查；`v3_compare_supplemental_checks.py` 使用实际 step200 模型和 Adam 状态验证下一次更新、mean/方差梯度隔离、49/50 步渐增及 Python/MATLAB 前向。补充数值验收输出 `supplemental_gpu_acceptance.json`。

`v3_compare_evaluation.py` 保存 564 个预测及 564 个 anchor；`v3_compare_local_audit.py` 在这些预测上重新核对带 case/subset 标识的 T02/T03/T04/V03 局部指标。T03 先找峰再匹配三条真实中心线；T04 采用原始十层剖面，10 µm 间距不作谷值分离判断。V03 是填充球体，采用连通域强度质心，合并连通域只能匹配一个微球。阈值 5/10/20% 分别保存。

局部评价规则和后处理源哈希保存在 `evaluation_rules.json` 和 `evaluation_source_snapshot/`。`analysis/` 中的局部表是最终权威版本，覆盖推理阶段的初步局部统计；原始预测未改动。用于统计形状误差的尺度对齐只影响指标，原始输出和正式物理评分不按 GT 重新拟合亮度。

`v3_compare_report.py` 与 `v3_compare_delivery.py` 生成中文报告、三次重复图册、固定位置剖面、训练/验证/梯度曲线、失败列表并核对全部原始文件哈希。最终状态见 `complete.json` 和 `final_acceptance.json`。

训练前曾有两次失败预检：一次 NumPy API 拼写错误，一次自设复现阈值比既定 1e-4 过严；均未启动训练并已归档到带 `failed_preflight` 后缀的独立目录。正式 run01 的三组均使用同一份通过预检的训练代码。
