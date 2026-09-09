# 2026-09-08 六卡续跑

用户授权把任务从物理 0 号 A40 扩展到 0–5 号六张 A40。此前已授权 0 号卡与
原有空闲计算上下文共存；1–5 号启动前必须无计算进程、利用率不高于 5%，全部卡
要求至少 24 GiB 可用。UUID 绑定后每个 MATLAB 工作进程只看见一张逻辑设备。

原始 SIMULATION_APPROVED.json、源代码快照、已存 prepared/simulation_config 的
单卡字段作为初始记录保留。当前资源授权来自根目录 EXECUTION_RESOURCES.json
所指的日期并行目录 authorization.json；只覆盖硬件调度，不改变数值协议。

## 切换与调度

- 先完整 CPU unittest 和 MATLAB 语法检查，通过后才切换。
- 仅对 PID/命令行均匹配本数据根目录的旧调度器与 MATLAB 进程发终止信号；
  最多等待一帧保存。其他人的进程不发送信号。
- 旧完整 MAT 结果逐文件指纹冻结。未发布的本任务 tp*.mat 临时文件若残留则移到
  并行目录 interrupted_temporary/ 可恢复保留；未知文件不清理。
- T04 尚未完成的整帧与整子集按代价平衡分为六组，各输出文件只由一个进程写入。
  不把单次 RL 的深度或迭代拆开，不做多卡归约，不改变原 solver。
- 同时冻结 V03 的 prepared/illumination，再用六路 CPU 原 ACC 前向生成 100 帧；
  不将 ACC 悄悄换成 GPU 近似。V03 散斑完成后在六张卡上并行完成整帧和子集 RL3。
- 每张卡先重算已有 T04 第 1 帧的 physics Mean，要求相对 L2 ≤1e-5，不覆盖参考
  文件。一致性记录在 progress/gpu*_parity.json，硬件与各组进度也单独保存。

输出仍进入原数据根目录；控制日志和代码快照另存
`parallel_dispatch_<YYYYMMDD>_runNN`（实际北京时间，同日递增）。
GPU 调度代码是新增文件，原始运行代码不修改，原代码指纹检查继续执行。

完成 T04/V03 后沿用 MATLAB 深度校验、AlgoRIM 输入导出、17 对象全文件哈希及
170 项读取验证，再发布最终清单。之前已完成重建的内容哈希必须不变。
只生成数据，不训练网络，不更换有 Set 基线，不改 RL3、PSF、真值和种子。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /workspace/xyx/.conda/envs/speckle_net/bin/python -m tools.run_dataset_v3_parallel \
  --root /workspace/xyx/speckle_illumination_net/data/speckle_dataset_v3_full_20260907_run01
```

`parallel_controller.json` 记录控制进程；根目录 `run_status.json` 指向并行目录，
后者 `progress/` 是各卡逐组进度，`processes/` 是本任务子进程的精确注册表。
出现错误时检查日志，不擅自覆盖不可变任务清单或更改数值条件。
