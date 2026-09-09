# 相机相关性测试扩展

使用 `tools/run_correlation_challenge.py` 为固定 V3 测试对象 T02/T03/T04
生成低／高相关各十帧。GPU 仅限物理 0 号，允许共享已有显存占用；少于
24 GiB 空闲显存则等待。不会使用 1–5 号卡，不训练或更改已有权重。

选帧仅读取浮点 `sensor_pre_detector`，完整视野逐帧空间去均值计算
Pearson。目标为 45 对绝对相关系数均值。固定 seed=20260908，1,000 个
唯一随机候选，低／高各 100 起点最优单帧交换至局部收敛；不宣称全局极值。
低／高组允许交集，均与原随机子集并列作为附加测试，不参与训练或选模。

资产存于数据集的 `test_challenges/correlation_<YYYYMMDD>_runNN`，通过
`datasets.correlation_challenge.load_challenge_input(manifest, sample_id, group)`
显式读取。原始采集图以只读使用的符号链接复用，清单保存其 SHA256；原始
数据清单和 30 项正式测试枚举不改。新 prepared.mat 为独立副本。

新输出一律用实际启动当天北京时间 `<实验名>_<YYYYMMDD>_runNN`，重复运行
递增编号。`--output` 仅用于恢复明确指定的已有运行，来源或权重哈希变化即
停止。先 CPU unittest，再 GPU RL3、网络推理，推理子进程退出后 CPU 报告。

主比较 final=400，best 沿用原无 GT 验证选择。Mean/Taylor-RL3 与三组网络
使用同一组帧。Taylor 保留 raw 和 sqrt，主比较 sqrt，网络不改变尺度。
GT 只用于评价。补集 90 帧是选帧后条件化的辅助物理评价，不是独立验证集。
低相关只是假设的信息冗余代理指标，不保证实际重建更好。
