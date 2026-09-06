#!/usr/bin/env python3
"""Evaluate frozen outputs; GT affects report metrics and displays only."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import tifffile
import torch


def matlab_volume(handle, key):
    return np.asarray(handle[key], dtype=np.float32).transpose(0, 2, 1).copy()


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def preserve_or_save_npy(path, volume):
    if path.exists():
        np.testing.assert_array_equal(np.load(path), volume)
    else:
        np.save(path, volume)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path)
    parser.add_argument('--require-subsets', action='store_true')
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    sys.path.insert(0, str(repo))
    from utils.reconstruction_metrics import reconstruction_metrics, _ssim_2d
    torch.set_num_threads(2)
    data = repo / 'data/matlab_cells_pilot_v2_r04'
    run = repo / 'outputs/multivolume_n10_no_mean_run01'
    report = args.report_dir.resolve() if args.report_dir else output / 'evaluation'
    if args.require_subsets:
        for scope in ['full100', 'subsets10']:
            manifest = json.loads((output / f'run_{scope}.json').read_text())
            assert manifest['status'] == 'complete' and manifest['source_files_unchanged'], scope
        assert len(list(output.glob('*/subset_*/mean/complete.json'))) == 30
        assert len(list(output.glob('*/subset_*/taylor/complete.json'))) == 30
    report.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(report / 'mpl_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows, proofs, subset_proofs, views = [], [], [], {}
    historical = {(r['sample_id'], int(r['subset_index'])): r
                  for r in csv.DictReader((run / 'test_metrics.csv').open())}
    z = np.arange(10, 101, 10)

    def measure(obj, method, frames, iteration, subset, pred, truth, source):
        assert pred.shape == truth.shape == (10, 260, 260)
        assert np.isfinite(pred).all() and np.all(pred >= 0)
        p, t = pred.astype(np.float64), truth.astype(np.float64)
        scale = float(np.sum(p * t) / max(np.sum(p * p), 1e-12))
        metrics = reconstruction_metrics(pred, truth, z)
        aligned = pred * scale
        metrics['gt_xy_mip_ssim_global_gain_aligned'] = _ssim_2d(truth.max(0), aligned.max(0))
        metrics['evaluation_only_global_gain'] = scale
        metrics['predicted_mass_at_truth_peak'] = float(pred.sum((1, 2))[
            np.argmax(truth.sum((1, 2)))] / max(float(pred.sum()), 1e-12))
        rows.append(dict(sample_id=obj, method=method, frames=frames, rl_iterations=iteration,
                         subset_index=subset, source=str(source), **metrics))
        if subset in (0, 1):
            views[(obj, method)] = (pred, aligned, truth)
        if method == 'Network10':
            for key in ['gt_scale_aligned_nrmse', 'gt_axial_w1_um', 'gt_support_outside_pm10_mass']:
                assert abs(metrics[key] - float(historical[(obj, subset)][key])) < 2e-7, key

    for obj in ['P07', 'T01', 'T02']:
        with h5py.File(data / obj / 'prepared.mat') as handle:
            truth = matlab_volume(handle, 'ground_truth')
        for subset in range(1, 11):
            subset_file = data / obj / 'subsets' / f'subset_{subset:02d}.mat'
            with h5py.File(subset_file) as handle:
                assert int(np.asarray(handle['iterations']).item()) == 3
                for method, key in [('Mean10_RL3', 'physics_mean_raw'),
                                    ('Taylor10_RL3', 'physics_taylor_sqrt_float')]:
                    measure(obj, method, 10, 3, subset, matlab_volume(handle, key), truth, subset_file)
            path = run / 'test/step_000160' / obj / f'subset_{subset:02d}/reconstruction.npy'
            measure(obj, 'Network10', 10, 3, subset, np.load(path), truth, path)
            with h5py.File(subset_file) as handle:
                indices = np.asarray(handle['input_indices']).ravel().astype(int).tolist()
                target_mean = np.asarray(handle['input_physics_mean_float']).T
                target_var = np.asarray(handle['input_physics_variance_nminus1_float']).T
            input_frames = []
            for index in indices:
                with h5py.File(data / obj / 'sensor_frames' / f'frame_{index:03d}.mat') as handle:
                    input_frames.append(np.asarray(handle['sensor_pre_detector'], dtype=np.float64).T)
            input_stack = np.stack(input_frames)
            np.testing.assert_allclose(input_stack.mean(0), target_mean, rtol=2e-13, atol=1e-14)
            np.testing.assert_allclose(input_stack.var(0, ddof=1), target_var, rtol=2e-13, atol=1e-14)
            for mode in ['mean', 'taylor']:
                folder = output / obj / f'subset_{subset:02d}' / mode
                if (folder / 'complete.json').is_file():
                    info = json.loads((folder / 'complete.json').read_text())
                    assert info['complete'] and info['iterations'] == 5 and info['frame_count'] == 10
                    assert info['sample_id'] == obj and info['subset_index'] == subset
                    assert info['method'] == mode and info['input_indices'] == indices
                    assert info['z_um'] == z.tolist()
                    with h5py.File(folder / 'reconstruction.mat') as handle:
                        pred = matlab_volume(handle, 'reconstruction')
                        raw = matlab_volume(handle, 'raw_solver_output')
                        np.testing.assert_array_equal(np.asarray(handle['sensor_mean']).T, target_mean)
                        np.testing.assert_array_equal(np.asarray(handle['sensor_variance_nminus1']).T, target_var)
                        expected = np.sqrt(raw) if mode == 'taylor' else raw
                        np.testing.assert_allclose(pred, expected, rtol=2e-7, atol=0)
                    np.testing.assert_array_equal(tifffile.imread(folder / 'reconstruction.tif'), pred)
                    preserve_or_save_npy(folder / 'reconstruction.npy', pred)
                    measure(obj, f'{mode.title()}10_RL5', 10, 5, subset,
                            pred, truth,
                            folder / 'reconstruction.tif')
                    subset_proofs.append(dict(sample_id=obj, method=mode, subset_index=subset,
                        input_indices_verified=True, statistics_recomputed_from_10_frames=True,
                        tiff_exact=True, shape=list(pred.shape),
                        positivity_fallback=info['diagnostic']['positivity_fallback_used']))
        # Independent verification of the 100-frame statistics and TIFF axis/scaling.
        frames = []
        for index in range(1, 101):
            with h5py.File(data / obj / 'sensor_frames' / f'frame_{index:03d}.mat') as handle:
                frames.append(np.asarray(handle['sensor_pre_detector'], dtype=np.float64).T)
        stack = np.stack(frames, axis=0)
        mean, var = stack.mean(0), stack.var(0, ddof=1)
        for mode in ['mean', 'taylor']:
            folder = output / obj / 'full100' / mode
            info = json.loads((folder / 'complete.json').read_text())
            assert info['complete'] and info['iterations'] == 5 and info['frame_count'] == 100
            assert info['input_indices'] == list(range(1, 101))
            path = folder / 'reconstruction.mat'
            with h5py.File(path) as handle:
                pred = matlab_volume(handle, 'reconstruction')
                raw = matlab_volume(handle, 'raw_solver_output')
                np.testing.assert_allclose(np.asarray(handle['sensor_mean']).T, mean, rtol=2e-13, atol=1e-14)
                np.testing.assert_allclose(np.asarray(handle['sensor_variance_nminus1']).T, var, rtol=2e-13, atol=1e-14)
                expected = np.sqrt(raw) if mode == 'taylor' else raw
                np.testing.assert_allclose(pred, expected, rtol=2e-7, atol=0)
            np.testing.assert_array_equal(tifffile.imread(folder / 'reconstruction.tif'), pred)
            preserve_or_save_npy(folder / 'reconstruction.npy', pred)
            measure(obj, f'{mode.title()}100_RL5', 100, 5, 0, pred, truth, path)
            proofs.append(dict(sample_id=obj, method=mode, shape=list(pred.shape),
                               statistics_verified=True, tiff_exact=True,
                               positivity_fallback=info['diagnostic']['positivity_fallback_used']))
    subset5_count = sum(row['method'].endswith('10_RL5') for row in rows)
    assert subset5_count in (0, 60), 'Wait for all 60 subset RL5 outputs before publishing a comparison'
    if args.require_subsets:
        assert subset5_count == 60 and len(rows) == 156
    write_csv(report / 'metrics_per_item.csv', rows)
    numeric = [key for key in rows[0] if key.startswith(('gt_', 'predicted_', 'truth_', 'evaluation_'))]
    methods = list(dict.fromkeys(row['method'] for row in rows))
    grouped = []
    for obj in ['P07', 'T01', 'T02']:
        for method in methods:
            selected = [row for row in rows if row['sample_id'] == obj and row['method'] == method]
            if not selected:
                continue
            result = dict(sample_id=obj, method=method, items=len(selected))
            for key in numeric:
                values = [row[key] for row in selected]
                result[key] = float(np.mean(values))
                # SD is descriptive variation across frozen input subsets, not object-level SE.
                result[key + '_subset_sd'] = float(np.std(values, ddof=0)) if len(values) > 1 else ''
            grouped.append(result)
    write_csv(report / 'metrics_per_object.csv', grouped)
    macro = []
    for method in methods:
        selected = [row for row in grouped if row['method'] == method]
        macro.append(dict(method=method, objects=len(selected),
                          **{key: float(np.mean([row[key] for row in selected])) for key in numeric}))
    write_csv(report / 'metrics_macro.csv', macro)
    (report / 'verification.json').write_text(json.dumps(dict(
        completed_full100_checks=proofs, previous_network_metrics_reproduced=True,
        completed_subsets10_checks=subset_proofs,
        gt_role='evaluation only; one scalar per volume, no per-layer gain or registration',
        subset_rl5_available=subset5_count,
        subset_rl5_expected=60), indent=2) + '\n')

    input_iteration = 5 if subset5_count == 60 else 3
    columns = ['GT', f'Mean10_RL{input_iteration}', f'Taylor10_RL{input_iteration}',
               'Network10', 'Mean100_RL5', 'Taylor100_RL5']
    fig, axes = plt.subplots(3, 6, figsize=(17, 9), constrained_layout=True)
    for i, obj in enumerate(['P07', 'T01', 'T02']):
        truth = views[(obj, 'Network10')][2]
        limit = float(truth.max())
        for j, method in enumerate(columns):
            volume = truth if method == 'GT' else views[(obj, method)][1]
            axes[i, j].imshow(volume.max(0), cmap='magma', vmin=0, vmax=limit)
            axes[i, j].set_title(f'{obj} | {method}', fontsize=10)
            axes[i, j].axis('off')
    fig.suptitle('Fixed subset 01 for N=10; one evaluation-only GT-aligned gain per full volume; shared color scale')
    fig.savefig(report / 'xy_comparison_subset01.png', dpi=150)
    # Common display transform only: retain the linear figure and numerical outputs.
    from matplotlib.colors import PowerNorm
    for i, obj in enumerate(['P07', 'T01', 'T02']):
        limit = float(views[(obj, 'Network10')][2].max())
        for axis in axes[i]:
            axis.images[0].set_norm(PowerNorm(gamma=0.4, vmin=0, vmax=limit))
    fig.suptitle('Fixed subset 01; GT-aligned whole-volume gain; identical display gamma=0.4 and color limits within each row')
    fig.savefig(report / 'xy_comparison_subset01_display_gamma04.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for axis, obj in zip(axes, ['P07', 'T01', 'T02']):
        truth = views[(obj, 'Network10')][2]
        mass = truth.sum((1, 2)); axis.plot(z, mass / mass.sum(), 'k-o', label='GT', linewidth=2)
        for method in columns[1:]:
            pred = views[(obj, method)][0]; mass = pred.sum((1, 2))
            axis.plot(z, mass / mass.sum(), '-o', label=method, markersize=3)
        axis.set(title=obj, xlabel='Depth (um)', ylabel='Fraction of total mass', ylim=(0, None))
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
    fig.savefig(report / 'axial_comparison_subset01.png', dpi=150); plt.close(fig)
    stage_note = ('All 60 N=10 RL5 controls are included.' if subset5_count == 60 else
                  'The 60 N=10 RL5 controls are a separate planned batch.')
    lines = ['# Test reconstruction comparison', '',
             'N=10 methods: per-subset metrics averaged within each object, then equally over 3 objects.',
             'N=100: one reconstruction per object. ' + stage_note,
             'MIP SSIM aligned below uses one evaluation-only GT-fitted gain per whole 3D volume.', '',
             '| Method | 3D aligned NRMSE | Axial W1 (um) | Support outside +/-10 | Aligned MIP SSIM |',
             '|---|---:|---:|---:|---:|']
    for row in macro:
        lines.append(f"| {row['method']} | {row['gt_scale_aligned_nrmse']:.4f} | "
                     f"{row['gt_axial_w1_um']:.3f} | {row['gt_support_outside_pm10_mass']:.2%} | "
                     f"{row['gt_xy_mip_ssim_global_gain_aligned']:.4f} |")
    lines += ['', 'Full detail: metrics_per_item.csv, metrics_per_object.csv, verification.json.',
              'Each N=10 figure uses subset_01, never an ensemble of ten input subsets.']
    (report / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
