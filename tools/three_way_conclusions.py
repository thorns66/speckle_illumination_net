"""Evidence-based Chinese interpretation of the completed, frozen comparison."""
import json
import numpy as np
from tools.three_way_experiment import OUTPUT

def write_report():
    from tools.three_way_report import csv_read
    analysis=OUTPUT/'analysis'
    summary=csv_read(analysis/'summary.csv');lookup={r['method']:r for r in summary}
    metrics=csv_read(analysis/'metrics.csv');brightness=csv_read(analysis/'brightness.csv')
    line=csv_read(analysis/'lines.csv');points=csv_read(analysis/'point_targets.csv')
    names={'baseline':'原 baseline','e1':'E1 亮度尺度','e2':'E2 相关方差','e3':'E3 均值与方差分工'}
    label=lambda method:names[method.split('_')[0]]+' '+method.split('_')[1]
    f=lambda row,key:float(row[key])
    pct=lambda value:f'{value:.1%}'
    rows=['# 三组实验：完整结果与判断','',
      '**三组均已完成 200 步训练，分别与原 baseline 的 best、final 比较。**',
      '结果有改善，也有明确退步。E1 解决了亮度不跟输入变化的问题；E2 修正了物理评分的错误偏好，但当前训练的形状质量没有全面改善；E3 的形状、深度和背景表现较好，但小断口更容易被连上。','',
      '本轮保留原条件：照明生成 NA=0.05，检测 PSF NA=0.15，无相机噪声。真实系统照明参数尚未确认，本报告不替代真实系统验证。三组从相同随机初始权重出发，训练种子 20260901，全局 batch=8，各 200 次更新。','',
      'best 仍按各组自己的验证 loss 选择；测试 GT 没有参与梯度更新或模型选择。final 使用第 200 步的实际权重。本次不根据测试结果临时更换正式 checkpoint。','',
      '## 1. 测试对象的整体结果','',
      '| 方法 | 权重步数 | 形状误差↓ | 深度分布误差 µm↓ | 目标横向范围外的输出占比↓ |',
      '|---|---:|---:|---:|---:|']
    for row in summary:
        rows.append(f"| {label(row['method'])} | {row['weight_step']} | {f(row,'test_aligned_nrmse'):.4f} | {f(row,'test_axial_w1_um'):.2f} | {pct(f(row,'test_background_xy_mass_fraction'))} |")
    rows+=['',
      '这里比较 P07、T01、T02 各十个输入子集。形状误差允许对整卷预测拟合一个显示/评价用的亮度系数；该系数没有乘回正式预测。深度分布误差是轴向 W1，既受位置影响，也受拖尾影响，**不能把它直接当成轴向分辨率或单点定位精度**。背景占比只统计 GT 横向范围外再留出两个像素缓冲的区域，深度拖尾另算。','',
      'E1 的整体形状和深度分布优于 baseline，但这里的背景占比没有降低。也就是说，**黑输入不再凭空发亮，和有物体时背景是否干净，是两件事。**','',
      'E2 的深度分布与背景有所改善，形状误差却高于 baseline。物理模型改正确是有价值的，但这套网络、学习率和 200 步训练安排，还没有把这种改正稳定转化为更好的整体形状。不能据此宣称相关项无用，也不能宣称 E2 已经成功替代 baseline。','',
      'E3 final 的形状和深度分布优于 E3 best，背景占比也进一步降低。它的测试均值评分却从 '+f"{f(lookup['e3_best'],'test_mean_loss'):.3f} 变为 {f(lookup['e3_final'],'test_mean_loss'):.3f}。这正是同时保留 final 的价值：loss 最优与结构最优仍不完全一致。",'',
      '## 2. 亮度修复是否真的有效','',
      '把三个对象的原始十帧分别变为 0.1、0.5、1、2 倍，再重新运行 MATLAB Mean-RL3 和 Taylor-RL3，最后交给冻结网络；另测全零输入。十二组输入的原始帧、均值、N−1 方差和 Taylor 单次开方均已复核，1 倍输入复现了原 RL3。','',
      '| 方法 | 完整流程最大缩放误差↓ | 黑输入最大值↓ |',
      '|---|---:|---:|']
    for row in summary:
        selected=[r for r in brightness if r['method']==row['method'] and float(r['gain']) in [.1,.5,2.]]
        error=max(float(r['relative_scale_error']) for r in selected)
        rows.append(f"| {label(row['method'])} | {error:.4%} | {f(row,'zero_max'):.6g} |")
    rows+=['',
      '缩放误差比较的是“缩放输入后的预测”与“原预测乘同一个倍数”，使用相对 L2，超过 100% 是可能的。E1 最大误差约 0.0064%，黑输入为零，说明这项修复在完整流程中成立。E2 仍不具备这种性质，不能靠修正方差前向自动解决输入亮度问题。','',
      'E3 也能较好跟随亮度，但原因包括依据十帧输入求整体增益。它验证的是“解析亮度校正 + 均值只调增益 + 方差只调结构”的整套方案，不能把改善全部归功于简单添加 mean loss。均值项到结构、方差项到亮度参数的梯度隔离已通过实际梯度检查。','',
      '![原始帧亮度变化与输出变化](analysis/brightness_response.png)','',
      '## 3. 点、线与断口有没有被改坏','',
      '主阈值固定为整卷最大值的 10%；完整结果同时保留 5%、20%。点先一对一匹配，再判断点对是否分开。下面的点对判据要求两个点都定位成功，且峰间谷值不超过弱峰的 80%。','',
      '| 方法 | 单点定位成功↑ | 横向点对分开↑ | 连续线假断口↓ | 已有断口被连上↓ |',
      '|---|---:|---:|---:|---:|']
    for row in summary:
        rows.append(f"| {label(row['method'])} | {pct(f(row,'single_localization_rate'))} | {pct(f(row,'lateral_pair_separation_rate'))} | {pct(f(row,'continuous_false_gap_rate'))} | {pct(f(row,'broken_bridge_rate'))} |")
    rows+=['',
      '每种方法的单点统计包含五个深度、三次采集、四个单点，共 60 次定位。baseline、E1、E3 都达到本次 ±2 像素、±10 µm 的定位标准；E2 best 和 final 分别有一次未通过。所有方法对 20、30、40 µm 的同 XY 轴向点对，分开率仍为 **0%**。目前没有证据说轴向双点分辨能力已经改善。','',
      '**连续线更完整，不代表断口也保留得更好。** E1、E3 的连续线假断口降到零，但原有小断口更容易被连上。按本次判据，主实验中出现的误连接全部来自 4 µm 断口；8、12 µm 未计出误连接。','',
      '| 方法 | 4 µm 断口误连接率↓ |', '|---|---:|']
    for row in summary:
        selected=[r for r in line if r['method']==row['method'] and float(r['threshold'])==.1 and float(r['gap_um'])==4]
        value=np.mean([r['bridged']=='True' for r in selected])
        rows.append(f"| {label(row['method'])} | {pct(value)} |")
    rows+=['',
      '因此，E1 可以保留为亮度处理的基础改进，但本次重新训练的 E1 并未提高横向点对分开率。E3 值得继续研究，同时必须把“小断口别被补平”列为下一轮需要解决的问题。当前不自动叠加三组改动，也不改原 baseline。','',
      '以下结构图只对每份**完整三维体**使用一个显示比例，所有层沿用该比例。原始预测不做这项归一化；同一绝对亮度比例的图另存为同名、不带 `_shape` 的文件。','',
      '![60 µm 线条结构比较](analysis/cases/lines_z060_r01_shape.png)',
      '![固定位置线条剖面](analysis/line_fixed_profiles.png)',
      '![各次采集分别显示的深度跟随](analysis/depth_following_all_repeats.png)','',
      '## 4. E2 是否真的修正了 loss 的错误偏好','',
      '这次补齐的是**同层横向相关与跨层相关**，没有把两个效应拆开做单独消融。计算时使用独立生成的照明统计库，不读取实际采集时的逐帧散斑。统计库仍依赖给定的光学模型，因此不能称为真实系统实测标定。','',
      '对 P08、P09、P10，每个对象使用三次新的独立九十帧统计。把正确 GT、加厚一次、加厚两次、上移一层、下移一层放在同一固定亮度单位下评分：','',
      '- 原方差 loss：9/9 次偏爱上移一层的错误结构。',
      '- 补齐相关项后：9/9 次在这五个候选中给正确 GT 最低分。',
      '- 在 0.25、0.5、1、2、4、8 倍这些预设增益档位中，原方差评分偏好 4 倍；修正方差及均值评分偏好 1 倍。没有为正式预测重新拟合增益。','',
      '这说明此前亮度冲突至少有一部分来自方差模型不完整，并非只需要调 mean loss 权重。但“正确 GT 排名更合理”与“当前训练结果变好”需要分别验收；本次 E2 的整体形状结果没有过后一关。','',
      '![固定 GT 增益扫描：三次重复分别绘制](analysis/fixed_gt_gain_audit.png)','',
      '各组 native loss 定义不同，不能用绝对大小跨组排名。共同均值评分、旧方差评分，以及独立照明库对三个诊断对象的评分，都保存在 CSV 中。','',
      '## 5. NA 参数偏差补充','']
    mismatch_path=analysis/'na_mismatch_metrics.csv'
    if mismatch_path.exists():
        mismatch=csv_read(mismatch_path)
        rows+=['使用点、线、轴向点对三个场景，各做一次配对采集。保持相同随机相位，把照明 NA 从 0.05 改成 0.04479，按固定孔径面积比补偿整体功率；冻结八套权重，重新做十帧 RL3 和网络重建。','',
          '| 方法 | 形状误差变化（NA 0.04479 减 NA 0.05） | 深度分布误差变化 µm |','|---|---:|---:|']
        for row in summary:
            selected=[r for r in mismatch if r['method']==row['method']]
            rows.append(f"| {label(row['method'])} | {np.mean([float(r['change_gt_scale_aligned_nrmse']) for r in selected]):+.4f} | {np.mean([float(r['change_gt_axial_w1_um']) for r in selected]):+.2f} |")
        rows+=['','本次小幅扰动没有出现整体重建突然失效，但只有一次配对重复，不能据此证明跨 NA 通用，更不能代替真实系统测量。它也没有回答照明 NA=0.15 时会怎样。']
    else:rows+=['配对重建补充尚未汇总；不据单纯换评分模型得出重建鲁棒性结论。']
    rows+=['','## 6. 怎样使用这次结果','',
      '优先保留 E1 的亮度尺度处理，因为完整流程已经验证它解决了明确问题；结构质量仍要单独把关。E2 的物理修正有直接证据支持，但当前 200 步训练结果不适合直接替代原 baseline。E3 的分工方案可以避免均值梯度直接拉动结构，final 显示出较好的形状表现，但 4 µm 断口粘连必须继续处理。','',
      '本轮只完成这三组独立验证，没有自动修改下一轮网络、loss 或 checkpoint 选择规则。每组只有一个训练随机种子；三次采集重复反映照明采样变化，不能代替多训练种子。原测试对象此前已用于分析，因此也不能称为从未接触过的全新测试集。','',
      '## 7. 文件与验收','',
      '- [总表](analysis/summary.csv)、[逐对象](analysis/per_object_summary.csv)、[三次重复](analysis/per_repeat_summary.csv)、[逐深度逐重复](analysis/per_depth_repeat_summary.csv)。',
      '- [点对间距/方向/亮度比](analysis/pair_resolution_summary.csv)、[断口/方向/亮度](analysis/line_quality_summary.csv)、[失败案例](analysis/failure_cases.csv)。',
      '- [所有点的固定位置剖面](analysis/depth_profiles.csv)、[所有线的固定位置剖面](analysis/line_fixed_profiles.csv)、[亮度逐项结果](analysis/brightness.csv)。',
      '- [固定结构 loss 排名](analysis/fixed_candidate_loss_audit.csv)、[固定 GT 增益扫描](analysis/fixed_gt_gain_audit.csv)、[新独立目标物理评分](analysis/covariance_audit.csv)、[NA 实际重建补充](analysis/na_mismatch_metrics.csv)。',
      '- 主预测：`evaluation/<baseline|e1|e2|e3>/<best|final>/<case>/reconstruction.npy`；同时保存 anchor。亮度预测在各自 `brightness/`，NA 补充在 `mismatch_evaluation/`。',
      '- 来源：[preflight.json](preflight.json)、[数值验收](gpu_checks.json)、[亮度输入复核](brightness_inputs/verification.json)、[最终文件验收](analysis/final_artifact_verification.json)、[执行约定](../../docs/three_way_validation_protocol_20260907.md)。','',
      '主评估共 912 份三维预测，亮度变化另有 120 份，NA 补充 24 份。原 baseline 权重与 37 个冻结源文件保持不变。新预测均要求尺寸正确、有限、非负，且哈希与所用 checkpoint 对应。','',
      '原 Python/MATLAB 前向相对 L2 为 5.15e-7；新增精确稀疏前向相对原 Python 前向为 6.61e-7。目标信息污染检查对八套权重分别执行，并与同输入重复计算对照；容差为相对 L2 1e-6。GPU 浮点累加可能带来约一个 FP32 ULP 的差异，实际误差保存于 `metrics.csv`。','',
      '恢复主流程：`python tools/run_three_way_validation.py --stage all --resume`。重建最终汇总与验收：`python tools/three_way_finish.py`。','',
      '![统一比较，包含改善和退步](analysis/comparison.png)']
    (OUTPUT/'REPORT_ZH.md').write_text('\n'.join(rows)+'\n')
