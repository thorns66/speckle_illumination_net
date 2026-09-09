# 17 对象正式数据构建

用户已批准 T03 分辨率板、T04 v2 轴向双层板（中心间距 10/20/30/40 μm）、
V03 60 颗微球，继续执行原数据计划。最新 GPU 授权覆盖预览阶段的“禁止共用”：
**仅物理 0 号 A40，允许与原有空闲计算上下文共存，不停止任何已有进程、不转卡。**
GPU 阶段前要求至少 24 GiB 空闲且利用率不高于 5%；身份/型号不符即停止。
CPU 正向仿真沿用原 ACC 算法；RL3 使用 UUID 绑定的 GPU 0，MATLAB 仅见逻辑卡 1。

## 划分与不变项

- 训练 P01–P11；P07/P09 回训练，旧 T01 的独立副本改名 P11。
- 验证 V01/V02/V03；测试 T02/T03/T04。整个对象的全部派生文件属于同一集合。
- 每个新对象 100 帧、10 个 10/90 子集、3 次 RL；物理及历史显示流均保留，
  保存 Taylor raw/sqrt、Mean 和单帧重建，导出 AlgoRIM 输入堆栈但不执行外部软件。
- 不启动网络训练，不替换 sqrt + Mean + Set + Gate 基线，不生成 RL5。

输出 `data/speckle_dataset_v3_full_<YYYYMMDD>_runNN` 使用实际启动的北京时间，
不覆盖历史数据或预览。14 个复用对象完整独立复制，复制后逐文件 SHA-256 核验。
迁移只改文本身份/划分/路径，MAT 重写后逐数组、种子、索引比对数值完全一致；
旧验证清单不沿用，重新计算全部 10/90 统计、检查重建/TIFF、重新建立文件哈希。
新对象的 truth.mat 是已批准预览的原样副本；其中预览 cfg 路径作为来源记录保留，
活动仿真配置在 prepared.mat/cfg 与 simulation_config.json，指向本次正式目录。

读取器按清单 version=2/3 选择明确划分；旧常量和旧配置不改。
全数据完成前没有正式完整清单；最终要通过 17 对象文件哈希和全部 170 个输入/目标
读取检查，再发布 FINAL_DATASET_MANIFEST.json 与 dataset_splits.json。

## 启动、状态与恢复

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /workspace/xyx/.conda/envs/speckle_net/bin/python -m tools.run_dataset_v3 --start
```

启动器建立日期目录和批准/源代码快照，再启动持久工作进程。`worker_launch.json`
记录 PID，`run_status.json` 是阶段状态，`active_child.json` 是当前 MATLAB 进程；
`progress/` 保存每帧/每子集进度，`logs/` 每次尝试独立编号。
异常写 failed 并停止，不擅自改计算条件。确认进程退出且原因解决后可用
`python -m tools.run_dataset_v3 --worker <绝对路径>` 显式恢复；代码指纹变化会拒绝恢复。
未完成的复制阶段不会被覆盖，必须检查后另建新日期 run。

10 μm 真值原生层相邻、无谷值采样，不能插值制造光学分离。新旧数据划分的实验
指标不能直接混算。本次报告只说明数据生成与验收，不声称网络性能提升。
