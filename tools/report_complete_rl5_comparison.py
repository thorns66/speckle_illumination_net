#!/usr/bin/env python3
"""Complete, non-cherry-picked report from the verified 156-item evaluation.

Only reporting: no checkpoint selection, registration, per-layer gain, or
ensemble prediction. ROI coordinates are frozen from the truth geometry.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

OBJECTS = ['P07', 'T01', 'T02']
METHODS = ['Mean10_RL5', 'Taylor10_RL5', 'Network10', 'Mean100_RL5', 'Taylor100_RL5']
METRICS = {
    'gt_scale_aligned_nrmse': ('3D 对齐 NRMSE↓', False),
    'gt_axial_w1_um': ('轴向 W1 µm↓', False),
    'gt_support_outside_pm10_mass': ('支持外质量↓', False),
    'gt_xy_mip_ssim_global_gain_aligned': ('对齐 MIP SSIM↑', True),
}
PITCH = 220 / 49 / 4
Z = np.arange(10, 101, 10)
# Same rectangle for every method/subset. Units: physical um, x0,x1,y0,y1.
ROIS = {'P07': (26, 133, 39, 128), 'T01': (198, 228, 23, 53),
        'T02': (105, 142, 51, 87)}
POINT = (91.14488086881906, 216.70397949574533, 60)


def read_csv(path):
    return list(csv.DictReader(path.open()))


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def apparent_fwhm(coordinates, values):
    """No background subtraction; linear half-height crossings, no extrapolation."""
    x, y = np.asarray(coordinates), np.asarray(values)
    peak = int(np.argmax(y))
    level = float(y[peak]) / 2
    if level <= 0 or peak == 0 or peak == len(y) - 1:
        return np.nan
    left = np.flatnonzero(y[:peak] <= level)
    right = np.flatnonzero(y[peak + 1:] <= level) + peak + 1
    if not len(left) or not len(right):
        return np.nan
    i, j = int(left[-1]), int(right[0])
    lo = x[i] + (level - y[i]) / (y[i + 1] - y[i]) * (x[i + 1] - x[i])
    hi = x[j - 1] + (level - y[j - 1]) / (y[j] - y[j - 1]) * (x[j] - x[j - 1])
    return float(hi - lo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    repo, output, report = args.repo.resolve(), args.output.resolve(), args.report_dir.resolve()
    items = read_csv(report / 'metrics_per_item.csv')
    groups = {(r['sample_id'], r['method']): r for r in read_csv(report / 'metrics_per_object.csv')}
    macro = {r['method']: r for r in read_csv(report / 'metrics_macro.csv')}
    lookup = {(r['sample_id'], r['method'], int(r['subset_index'])): r for r in items}
    proof = json.loads((report / 'verification.json').read_text())
    assert len(items) == 156 and proof['subset_rl5_available'] == 60
    assert len(proof['completed_subsets10_checks']) == 60
    assert proof['previous_network_metrics_reproduced']
    figures = report / 'figures'; figures.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(report / 'mpl_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    training_run = repo / 'outputs/multivolume_n10_no_mean_run01'
    training = read_csv(training_run / 'training_metrics.csv')
    validation = [dict(step=int(p.stem.rsplit('_', 1)[1]), **json.loads(p.read_text()))
                  for p in sorted(training_run.glob('validation_step_*.json'))]
    completion = json.loads((training_run / 'training_complete.json').read_text())
    assert completion['best_step'] == 160 and completion['completed_steps'] == 200
    write_csv(report / 'training_validation_history.csv', validation)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.7), constrained_layout=True)
    axes[0].plot([int(r['step']) for r in training], [float(r['total_loss']) for r in training],
                 color='gray', alpha=.45, lw=.7, label='Training batch loss')
    steps = [r['step'] for r in validation]
    axes[0].plot(steps, [r['selection_score'] for r in validation], 'o-', label='Validation physical loss')
    axes[0].set(title='Training / validation objective', ylabel='Physical loss'); axes[0].legend(fontsize=7)
    for axis, key, title in [(axes[1], 'gt_scale_aligned_nrmse', 'Validation GT-aligned NRMSE'),
                             (axes[2], 'gt_axial_w1_um', 'Validation axial W1 (um)')]:
        axis.plot(steps, [r[key] for r in validation], 'o-'); axis.set_title(title)
    for axis in axes:
        axis.axvline(160, color='red', ls='--', lw=1); axis.set_xlabel('Optimizer step'); axis.grid(alpha=.2)
    fig.suptitle('Checkpoint selected by physical validation loss only; GT curves are diagnostic')
    fig.savefig(figures / 'training_validation_history.png', dpi=150); plt.close(fig)

    truths, volumes, gains = {}, {}, {}
    provenance_paths = [training_run / 'checkpoint_best.pt', training_run / 'config_used.yaml',
                        Path(__file__).resolve(), repo / 'tools/evaluate_test_rl5_baselines.py']
    baseline_metadata = [json.loads(p.read_text()) for p in output.glob('*/full100/*/complete.json')]
    baseline_metadata += [json.loads(p.read_text()) for p in output.glob('*/subset_*/*/complete.json')]
    assert len(baseline_metadata) == 66
    psf_sources = {r['psf_source'] for r in baseline_metadata}
    assert len(psf_sources) == 1
    assert all(r['selected_psf_indices_one_based'] == list(range(1, 11)) for r in baseline_metadata)
    provenance_paths += [Path(p) for p in psf_sources]
    for obj in OBJECTS:
        provenance_paths.append(repo / 'data/matlab_cells_pilot_v2_r04' / obj / 'prepared.mat')
        with h5py.File(repo / 'data/matlab_cells_pilot_v2_r04' / obj / 'prepared.mat') as handle:
            truths[obj] = np.asarray(handle['ground_truth'], dtype=np.float32).transpose(0, 2, 1)
        for method in METHODS:
            for subset in ([0] if '100' in method else range(1, 11)):
                row = lookup[(obj, method, subset)]
                if method == 'Network10':
                    path = Path(row['source'])
                else:
                    tag = 'full100' if subset == 0 else f'subset_{subset:02d}'
                    path = output / obj / tag / method.split('10')[0].lower() / 'reconstruction.npy'
                pred = np.load(path)
                provenance_paths.append(path)
                assert pred.shape == truths[obj].shape and np.isfinite(pred).all()
                volumes[obj, method, subset] = pred
                gains[obj, method, subset] = float(row['evaluation_only_global_gain'])

    def volume(obj, method, subset, aligned=True):
        if method == 'GT':
            return truths[obj]
        key = (obj, method, 0 if '100' in method else subset)
        return volumes[key] * gains[key] if aligned else volumes[key]

    pairs, pair_summaries = [], []
    comparisons = [('Network10', 'Mean10_RL5'), ('Network10', 'Taylor10_RL5'),
                   ('Network10', 'Mean10_RL3'), ('Network10', 'Taylor10_RL3'),
                   ('Mean10_RL5', 'Mean10_RL3'), ('Taylor10_RL5', 'Taylor10_RL3')]
    for target, baseline in comparisons:
        for obj in OBJECTS:
            for metric, (_, higher) in METRICS.items():
                deltas = []
                for subset in range(1, 11):
                    a = float(lookup[obj, target, subset][metric])
                    b = float(lookup[obj, baseline, subset][metric])
                    benefit = a - b if higher else b - a
                    deltas.append(benefit)
                    pairs.append(dict(sample_id=obj, subset_index=subset, target=target,
                                      baseline=baseline, metric=metric, target_value=a,
                                      baseline_value=b, benefit_positive_is_better=benefit))
                pair_summaries.append(dict(sample_id=obj, target=target, baseline=baseline,
                    metric=metric, mean_benefit=float(np.mean(deltas)),
                    subset_sd=float(np.std(deltas)), wins=int(np.sum(np.asarray(deltas) > 1e-10)),
                    ties=int(np.sum(np.abs(deltas) <= 1e-10)), paired_subsets=10))
    write_csv(report / 'paired_differences.csv', pairs)
    write_csv(report / 'paired_summary.csv', pair_summaries)
    full_comparisons = []
    for obj in OBJECTS:
        for baseline in METHODS[-2:]:
            for metric, (_, higher) in METRICS.items():
                a, b = float(groups[obj, 'Network10'][metric]), float(groups[obj, baseline][metric])
                full_comparisons.append(dict(sample_id=obj, baseline=baseline, metric=metric,
                    network_10_subset_mean=a, baseline_100_single=b,
                    benefit_positive_is_better=a - b if higher else b - a))
    write_csv(report / 'network_vs_full100_per_object.csv', full_comparisons)

    columns = ['GT'] + METHODS
    # Show all subsets, not just favorable examples; no averaging of predictions.
    for subset in range(1, 11):
        fig, axes = plt.subplots(3, 6, figsize=(15, 8), constrained_layout=True)
        for i, obj in enumerate(OBJECTS):
            norm = PowerNorm(.4, vmin=0, vmax=float(truths[obj].max()))
            for j, method in enumerate(columns):
                axes[i, j].imshow(volume(obj, method, subset).max(0), cmap='magma', norm=norm)
                axes[i, j].set_title(f'{obj} | {method}', fontsize=9)
                axes[i, j].axis('off')
        fig.suptitle(f'Fixed subset {subset:02d}; evaluation-only whole-volume gain; shared gamma=0.4 per row')
        fig.savefig(figures / f'xy_subset_{subset:02d}.png', dpi=135)
        if subset == 1:
            fig.savefig(report / 'preview_xy.png', dpi=65)
            fig.savefig(report / 'preview_xy.jpg', dpi=75, pil_kwargs={'quality': 80, 'optimize': True})
        plt.close(fig)

    layer_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    for axis, obj in zip(axes, OBJECTS):
        mass = truths[obj].sum((1, 2)); truth_fraction = mass / mass.sum()
        axis.plot(Z, truth_fraction, 'k-o', label='GT', lw=2)
        for depth, fraction in zip(Z, truth_fraction):
            layer_rows.append(dict(sample_id=obj, method='GT', subset_index=0,
                                   depth_um=int(depth), mass_fraction=float(fraction)))
        for method in METHODS:
            fractions = []
            for subset in ([0] if '100' in method else range(1, 11)):
                mass = volumes[obj, method, subset].astype(np.float64).sum((1, 2))
                fraction = mass / mass.sum(); fractions.append(fraction)
                for depth, value in zip(Z, fraction):
                    layer_rows.append(dict(sample_id=obj, method=method, subset_index=subset,
                                           depth_um=int(depth), mass_fraction=float(value)))
            fractions = np.stack(fractions)
            mean, sd = fractions.mean(0), fractions.std(0)
            line, = axis.plot(Z, mean, '-o', ms=3, label=method)
            if len(fractions) > 1:
                axis.fill_between(Z, np.maximum(0, mean - sd), mean + sd,
                                  color=line.get_color(), alpha=.12)
        axis.set(title=obj, xlabel='Depth (um)', ylabel='Fraction of total 3D mass', ylim=(0, 1.02))
        axis.grid(alpha=.2); axis.legend(fontsize=7)
    fig.suptitle('N=10: mean +/- descriptive subset SD of normalized mass, NOT an ensemble reconstruction')
    fig.savefig(figures / 'axial_all_subsets.png', dpi=150); plt.close(fig)
    write_csv(report / 'axial_mass_per_layer.csv', layer_rows)

    roi_rows, profile_rows, width_rows = [], [], []
    for obj in OBJECTS:
        norm = PowerNorm(.4, vmin=0, vmax=float(truths[obj].max()))
        fig, axes = plt.subplots(2, 6, figsize=(15, 5.5), constrained_layout=True)
        for j, method in enumerate(columns):
            pred = volume(obj, method, 1)
            for i, projection in enumerate([pred.max(1), pred.max(2)]):
                axes[i, j].imshow(projection, cmap='magma', norm=norm, origin='lower',
                                  aspect='auto', extent=(-PITCH / 2, 259.5 * PITCH, 5, 105))
                axes[i, j].set_title(f'{method} | {"XZ" if i == 0 else "YZ"}', fontsize=9)
                axes[i, j].set_xlabel('x (um)' if i == 0 else 'y (um)')
                if j == 0: axes[i, j].set_ylabel('z (um)')
        fig.suptitle(f'{obj}, subset 01; shared whole-volume gain/color scale; z sampling = 10 um')
        fig.savefig(figures / f'{obj}_orthogonal_subset01.png', dpi=150); plt.close(fig)
        fig, axes = plt.subplots(10, 6, figsize=(12, 19), constrained_layout=True)
        for j, method in enumerate(columns):
            pred = volume(obj, method, 1)
            for layer in range(10):
                axes[layer, j].imshow(pred[layer], cmap='magma', norm=norm)
                axes[layer, j].set_title(f'{method}, z={Z[layer]}', fontsize=8)
                axes[layer, j].axis('off')
        fig.suptitle(f'{obj}, subset 01: every depth; no layer-wise normalization')
        fig.savefig(figures / f'{obj}_all_depths_subset01.png', dpi=110); plt.close(fig)
        x0, x1, y0, y1 = ROIS[obj]
        xs = np.flatnonzero((np.arange(260) * PITCH >= x0) & (np.arange(260) * PITCH <= x1))
        ys = np.flatnonzero((np.arange(260) * PITCH >= y0) & (np.arange(260) * PITCH <= y1))
        crop = (slice(None), slice(ys[0], ys[-1] + 1), slice(xs[0], xs[-1] + 1))
        fig, axes = plt.subplots(1, 6, figsize=(15, 3), constrained_layout=True)
        for axis, method in zip(axes, columns):
            axis.imshow(volume(obj, method, 1)[crop].max(0), cmap='magma', norm=norm,
                        extent=(xs[0] * PITCH, xs[-1] * PITCH, ys[-1] * PITCH, ys[0] * PITCH))
            axis.set_title(method, fontsize=9); axis.set_xlabel('x (um)')
        axes[0].set_ylabel('y (um)')
        fig.suptitle(f'{obj}: fixed truth-geometry ROI, subset 01; shared color scale')
        fig.savefig(figures / f'{obj}_fixed_roi_subset01.png', dpi=150); plt.close(fig)
        for method in METHODS:
            for subset in ([0] if '100' in method else range(1, 11)):
                pred, truth = volume(obj, method, subset)[crop], truths[obj][crop]
                roi_rows.append(dict(sample_id=obj, method=method, subset_index=subset,
                    global_gain_aligned_roi_nrmse=float(np.linalg.norm(pred - truth) / np.linalg.norm(truth)),
                    roi_x0_um=x0, roi_x1_um=x1, roi_y0_um=y0, roi_y1_um=y1))
    write_csv(report / 'fixed_roi_metrics.csv', roi_rows)

    # T01 first isolated point, fixed BEFORE examining RL5: true z=60, no peak-layer selection.
    px, py, pz = POINT; ix, iy, iz = round(px / PITCH), round(py / PITCH), list(Z).index(pz)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for dim, axis in zip(['x', 'y'], axes):
        center = px if dim == 'x' else py
        indices = np.flatnonzero(abs(np.arange(260) * PITCH - center) <= 8)
        positions = indices * PITCH - center
        for method in columns:
            for subset in ([0] if method == 'GT' or '100' in method else range(1, 11)):
                pred = volume('T01', method, subset)
                values = pred[iz, iy, indices] if dim == 'x' else pred[iz, indices, ix]
                width = apparent_fwhm(positions, values)
                width_rows.append(dict(sample_id='T01', method=method, subset_index=subset,
                    axis=dim, true_z_um=pz, apparent_fwhm_um=width,
                    both_half_height_crossings=bool(np.isfinite(width)),
                    peak_offset_um=float(positions[int(np.argmax(values))])))
                for position, value in zip(positions, values):
                    profile_rows.append(dict(method=method, subset_index=subset, axis=dim,
                                             offset_um=float(position), aligned_intensity=float(value)))
                if subset in (0, 1):
                    axis.plot(positions, values, '-o', ms=3, label=method)
        axis.set(xlabel=f'{dim} offset from truth point (um)', ylabel='Globally aligned intensity')
        axis.grid(alpha=.2); axis.legend(fontsize=7)
    fig.suptitle('T01 first isolated point at true z=60 um; subset 01; no local normalization')
    fig.savefig(figures / 'T01_point_profiles_subset01.png', dpi=150); plt.close(fig)
    write_csv(report / 'T01_point_profiles_all_subsets.csv', profile_rows)
    write_csv(report / 'T01_point_apparent_fwhm.csv', width_rows)
    roi_summary = []
    for obj in OBJECTS:
        for method in METHODS:
            values = [r['global_gain_aligned_roi_nrmse'] for r in roi_rows
                      if r['sample_id'] == obj and r['method'] == method]
            roi_summary.append(dict(sample_id=obj, method=method, items=len(values),
                                    roi_nrmse_mean=float(np.mean(values)),
                                    roi_nrmse_subset_sd=float(np.std(values)) if len(values)>1 else ''))
    write_csv(report / 'fixed_roi_summary.csv', roi_summary)
    width_summary = []
    for method in columns:
        for dim in ['x', 'y']:
            selected = [r for r in width_rows if r['method'] == method and r['axis'] == dim]
            values = [r['apparent_fwhm_um'] for r in selected if r['both_half_height_crossings']]
            width_summary.append(dict(method=method, axis=dim, attempted=len(selected),
                measured=len(values), apparent_fwhm_mean_um=float(np.mean(values)) if values else np.nan,
                mean_absolute_peak_offset_um=float(np.mean([abs(r['peak_offset_um']) for r in selected]))))
    write_csv(report / 'T01_point_apparent_fwhm_summary.csv', width_summary)
    assert len(pairs) == 720 and len(pair_summaries) == 72 and len(full_comparisons) == 24
    assert len(roi_rows) == 96 and len(layer_rows) == 990 and len(width_rows) == 66
    source_hashes = {}
    for path in provenance_paths:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        source_hashes[str(path)] = digest.hexdigest()
    (report / 'report_source_sha256.json').write_text(json.dumps(source_hashes, indent=2) + '\n')

    def fmt(metric, value):
        value = float(value)
        return f'{value:.2%}' if metric == 'gt_support_outside_pm10_mass' else f'{value:.4f}'

    headline_metrics = list(METRICS)[:3]
    same_frame_gain = all(float(macro['Network10'][k]) < float(macro[baseline][k])
                         for baseline in METHODS[:2] for k in headline_metrics)
    nrmse_failures = [f"{obj} 相对 {method}（网络 {float(groups[obj, 'Network10']['gt_scale_aligned_nrmse']):.4f}，基线 {float(groups[obj, method]['gt_scale_aligned_nrmse']):.4f}）"
                     for obj in OBJECTS for method in METHODS if method != 'Network10'
                     and float(groups[obj, 'Network10']['gt_scale_aligned_nrmse']) >= float(groups[obj, method]['gt_scale_aligned_nrmse'])]
    takeaway = ('网络在三维对齐 NRMSE、轴向 W1、支持外质量这三个宏平均指标上，均优于同帧数的 Mean10-RL5 和 Taylor10-RL5。'
                if same_frame_gain else '网络与同帧数 RL5 的优劣存在指标取舍，不能声称三个宏平均指标全部改善。')
    exceptions = ('对象级三维误差未胜出的情况：' + '；'.join(nrmse_failures) + '。'
                  if nrmse_failures else '对象级三维对齐误差也均更低，但仍需单独检查局部结构与错层。')

    lines = ['# 测试集完整重建对比（RL5）', '',
        '已补齐 60/60 个 10 帧 RL5 重建体：3 个测试对象 × 10 个固定子集 × Mean/Taylor 两种方法。',
        '主比较包含 60 个新 RL5、30 个既有网络输出、6 个 100 帧 RL5；另保留 60 个 RL3 控制，共 156 条逐体指标。', '',
        '## 主要结论', '', takeaway, '', exceptions, '',
        '相对基线提升不等于已高质量恢复。T01/T02 的网络深度峰值、质心及支持外质量见对象级表；必须同时检查轴向扩散、细丝展宽和局部定位，而不能只看稀疏背景占比很高的 MIP SSIM。', '',
        '固定局部 ROI 也存在反例：T02 断口区域的三维 NRMSE，网络为 0.9819，Mean10-RL5 为 0.9392，Taylor10-RL5 为 0.9524。网络的整幅指标改善并未转化为这个局部区域的改善。', '',
        '人工检查预先固定的 subset 01 图：网络的 XY 背景较少，但 P07 细丝仍明显粘连；T01 固定点对仍合并为团块；T02 线条存在斑点/串珠状纹理，断口两端仍膨胀成块状。因此本轮支持“宏平均结构/轴向指标改善”，不支持“所有局部结构或光学分辨率全面提升”。', '',
        '## 上次训练情况', '',
        '8 个训练物体、3 个验证物体、3 个测试物体；10 帧输入，互补 90 帧用于训练方差约束。完成 200 个优化步，batch=8，六卡训练；不是 200 个 epoch。',
        '模型启用 Mean、VAR、Set(mean/std) 和 gate。`no_mean` 指 Mean 损失权重为零，不是删除 Mean 输入分支；训练主要优化方差物理损失，TV 权重 1e-5。',
        '验证物理损失在 step 160 最低：0.280555；step 200 为 0.359526。既定选择规则因此使用 step 160，测试物理损失为 0.409853。',
        '值得注意：验证集对齐 NRMSE 从 step 160 的 0.925324 继续降到 step 200 的 0.909986，W1 从 14.0907 降到 13.4748 µm，而物理损失反而升高。这说明物理损失与真值重建质量并非严格同向；不能仅凭 loss 判断质量，也不在这次测试后临时换 checkpoint。', '',
        '![Training history](figures/training_validation_history.png)', '',
        '## 统一口径', '',
        '- 测试对象 P07/T01/T02；网络固定 `multivolume_n10_no_mean_run01` 的 step 160，训练共 200 步，按验证集物理损失选最优，未用测试真值选 checkpoint。',
        '- 网络仍使用原来的 10 帧 RL3 输入；没有改成 RL5 输入，也没有重训。每个子集与 Mean10/Taylor10 严格使用相同原始 10 帧。',
        '- Mean 为传感器均值经 H/Ht 五次重建；Taylor 为 N−1 样本方差经 H²/Ht² 五次重建后开方一次。沿用项目历史 `X *= Hty / Ht(HX)` 更新，不宣称等同于所有标准 RL 实现。',
        '- 10 帧方法先在每个物体的 10 子集上汇总，再对 3 个物体等权平均。独立物体数为 3，不是 30；子集 SD 和胜率仅描述稳定性，不作独立样本显著性推断。',
        '- 100 帧方法每个物体只有一份输出，是多帧参考，不是同帧公平对比，也不是真值或理论上限。',
        '- 全局对齐只在评价时拟合一个三维体积增益；原始重建不变，不逐层缩放、不配准。原尺度 NRMSE、SSIM、增益均保存在 CSV。SSIM 沿用项目 11×11 box-window 实现及成对动态范围，稀疏背景会提高分数，不能据此单独声称分辨率提升。', '',
        '## 三个物体等权宏平均', '',
        '| 方法 | ' + ' | '.join(label for label, _ in METRICS.values()) + ' |',
        '|---|---:|---:|---:|---:|']
    for method in METHODS:
        lines.append('| ' + method + ' | ' + ' | '.join(fmt(key, macro[method][key]) for key in METRICS) + ' |')
    lines += ['', '### 网络相对基线的变化', '',
              'NRMSE/W1 为相对下降率；支持外质量为百分点减少；负数表示网络更差。', '',
              '| 基线 | NRMSE 下降 | W1 下降 | 支持外质量减少（百分点） |', '|---|---:|---:|---:|']
    for method in METHODS:
        if method == 'Network10': continue
        base, net = macro[method], macro['Network10']
        decreases = [(float(base[k]) - float(net[k])) / float(base[k]) for k in list(METRICS)[:2]]
        mass_delta = 100 * (float(base['gt_support_outside_pm10_mass']) - float(net['gt_support_outside_pm10_mass']))
        lines.append(f'| {method} | {decreases[0]:.2%} | {decreases[1]:.2%} | {mass_delta:.2f} |')
    for obj in OBJECTS:
        lines += ['', f'## {obj}：对象级结果', '',
                  '10 帧方法为均值 ± 子集 SD；100 帧只有单个结果。', '',
                  '| 方法 | 3D 对齐 NRMSE↓ | W1 µm↓ | 支持外质量↓ | 对齐 MIP SSIM↑ |', '|---|---:|---:|---:|---:|']
        for method in METHODS:
            row = groups[obj, method]
            cells = [fmt(k, row[k]) + (f" ± {fmt(k, row[k + '_subset_sd'])}" if '100' not in method else '') for k in METRICS]
            lines.append('| ' + method + ' | ' + ' | '.join(cells) + ' |')
        lines += ['', '网络对同帧 RL5 的逐子集胜数（仅描述性，分母为 10）：', '',
                  '| 基线 | NRMSE | W1 | 支持外质量 | MIP SSIM |', '|---|---:|---:|---:|---:|']
        for baseline in METHODS[:2]:
            wins = [next(r['wins'] for r in pair_summaries if r['sample_id'] == obj and r['target'] == 'Network10' and r['baseline'] == baseline and r['metric'] == k) for k in METRICS]
            lines.append('| ' + baseline + ' | ' + ' | '.join(f'{v}/10' for v in wins) + ' |')
        netrows = [lookup[obj, 'Network10', subset] for subset in range(1, 11)]
        peaks = sorted(set(float(r['predicted_depth_peak_um']) for r in netrows))
        lines += ['', f"网络峰值层：{peaks} µm；平均轴向质心 {float(groups[obj, 'Network10']['predicted_depth_centroid_um']):.2f} µm；真值质心 {float(groups[obj, 'Network10']['truth_depth_centroid_um']):.2f} µm。",
                  '', f'![{obj} XZ/YZ](figures/{obj}_orthogonal_subset01.png)',
                  '', f'![{obj} fixed ROI](figures/{obj}_fixed_roi_subset01.png)']
    lines += ['', '## RL3 补充控制', '', '| 方法 | 3D 对齐 NRMSE↓ | W1 µm↓ | 支持外质量↓ | 对齐 MIP SSIM↑ |', '|---|---:|---:|---:|---:|']
    for method in ['Mean10_RL3', 'Taylor10_RL3']:
        lines.append('| ' + method + ' | ' + ' | '.join(fmt(k, macro[method][k]) for k in METRICS) + ' |')
    lines += ['', '## 固定局部 ROI 的三维误差', '',
              '继续沿用全局增益，不对 ROI 重新拟合。低误差不代表已分辨所有局部细节。', '',
              '| 方法 | P07 ROI NRMSE↓ | T01 ROI NRMSE↓ | T02 ROI NRMSE↓ |', '|---|---:|---:|---:|']
    for method in METHODS:
        values = [next(r['roi_nrmse_mean'] for r in roi_summary if r['sample_id']==obj and r['method']==method) for obj in OBJECTS]
        lines.append('| ' + method + ' | ' + ' | '.join(f'{v:.4f}' for v in values) + ' |')
    lines += ['', '### T01 固定孤立点附近的横向剖面', '',
              '只在真实 z=60 µm 测量。这里的表观 FWHM 是固定窗口内最大响应的半高宽，最大响应不保证属于目标点。只对两侧交点都存在的剖面平均，括号为有效数/总数；缺交点不补造数值。',
              '特别注意：网络窗口最大响应的 x/y 平均绝对偏移为 4.49/5.04 µm，x/y 仅 4/10、6/10 个剖面能测到两侧半高交点。这表明该点未可靠定位；约 1.12/1.81 µm 的窄响应可能属于偏位伪峰，不能解释为目标点 FWHM 或分辨率超越真值。', '',
              '| 方法 | x 表观 FWHM µm（有效数） | y 表观 FWHM µm（有效数） | x/y 平均绝对峰偏移 µm |', '|---|---:|---:|---:|']
    for method in columns:
        rows_xy = [next(r for r in width_summary if r['method']==method and r['axis']==dim) for dim in ['x','y']]
        cells = [(f"{r['apparent_fwhm_mean_um']:.3f}" if r['measured'] else '未测得') + f" ({r['measured']}/{r['attempted']})" for r in rows_xy]
        shifts = '/'.join(f"{r['mean_absolute_peak_offset_um']:.3f}" for r in rows_xy)
        lines.append('| ' + method + ' | ' + ' | '.join(cells) + ' | ' + shifts + ' |')
    lines += ['', 'RL3→RL5 与 RL3→Network 的逐体差值均见 paired_differences.csv；不能仅凭超过 RL3 就声称优于 RL5。', '',
              '## 完整图像与审计', '',
              '![XY subset 01](figures/xy_subset_01.png)', '',
              '![Axial all subsets](figures/axial_all_subsets.png)', '',
              '![T01 point profiles](figures/T01_point_profiles_subset01.png)', '',
              '- `figures/xy_subset_01.png` 至 `xy_subset_10.png` 覆盖全部 10 个子集；每份图都是一次 10 帧输出，未将十份网络结果平均冒充 10 帧。',
              '- 每对象包含 XZ/YZ、全部 10 深度层、预先按真值几何固定的 ROI；色标在同对象内统一，显示 gamma=0.4，线性图保留在 xy_comparison_subset01.png。',
              '- `fixed_roi_metrics.csv` 使用既定全局增益，不在局部重新拟合；ROI 并非从预测中挑选。',
              '- T01 点剖面固定第一个孤立点与真值 z=60 µm，x/y 半窗 8 µm。FWHM 不扣背景、按两侧半高线性插值，缺任一交点记 NaN，不能当作已测得光学分辨率；轴向采样仍为 10 µm。',
              '- `metrics_per_item.csv`、`metrics_per_object.csv`、`metrics_macro.csv` 为完整数值；`paired_summary.csv` 为配对增益/胜数；`network_vs_full100_per_object.csv` 不把单个 100 帧结果复制为多个独立样本。',
              '- `verification.json` 核验冻结帧索引、重新计算传感器统计量、Taylor 开方语义、TIFF/MAT 逐元素一致和既有网络指标复现。', '',
              '## 结论边界', '',
              '这些结果仅代表 3 个已查看的无额外相机噪声仿真测试对象，以及固定五次迭代的历史基线。宏平均改善不等于每个对象或每个局部结构都改善。充分调优的传统方法需在验证集选迭代数；根据本次测试结果继续改网络后，正式泛化验证应使用新测试对象。',
              '本轮重建运行存在共享 GPU 和多批 PSF 加载；保存的 diagnostic 耗时可查，但不作为公平速度排名。网络的 200 步训练时间也不是单体推理时间。']
    lines += ['', '## 可点击产物索引', '',
              '[逐体指标](metrics_per_item.csv) · [对象汇总](metrics_per_object.csv) · [宏平均](metrics_macro.csv) · [配对统计](paired_summary.csv)', '',
              '全部固定子集 XY 对比：' + ' · '.join(f'[subset {s:02d}](figures/xy_subset_{s:02d}.png)' for s in range(1, 11)), '',
              '逐层完整图：' + ' · '.join(f'[{obj} 全部深度](figures/{obj}_all_depths_subset01.png)' for obj in OBJECTS), '',
              '[固定 ROI 数值](fixed_roi_summary.csv) · [点剖面及有效测量数](T01_point_apparent_fwhm_summary.csv) · [来源哈希](report_source_sha256.json)']
    (report / 'report_zh.md').write_text('\n'.join(lines) + '\n')
    (report / 'report_contract.json').write_text(json.dumps(dict(
        complete=True, metric_rows=len(items), objects=OBJECTS, independent_objects=3,
        required_subset_rl5=60, primary_reconstructions=96, all_xy_subsets=list(range(1, 11)),
        roi_bounds_um=ROIS, point_xyz_um=POINT, object_pitch_um=PITCH,
        display_gamma=.4, gain='one evaluation-only scalar per original 3D volume',
        no_prediction_ensemble=True, numerical_fwhm_missing_policy='NaN, never extrapolate'), indent=2) + '\n')
    print(f'COMPLETE: {report / "report_zh.md"}; {len(list(figures.glob("*.png")))} figures')


if __name__ == '__main__':
    main()
