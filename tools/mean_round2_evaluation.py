"""Read-only model evaluation and isolated reporting for the second mean round.

The original acquisition, GT, reference predictions and reports are never changed.
Only artifacts below OUTPUT are written. Run evaluate in a GPU worker; report is CPU.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for _search_path in (str(_REPO), str(_REPO / "tools")):
    if _search_path not in sys.path:
        sys.path.insert(0, _search_path)

import numpy as np
import torch

from tools.three_way_experiment import ROOT, DATA, PRIORITY, install_data, sha256, write_json
from tools.three_way_report import (
    OUTPUT as OLD_OUTPUT, all_cases, input_item, targets, brightness_item,
    csv_read, csv_write,
)
from tools.priority_validation_common import load_config
import tools.priority_validation_analysis as pa
from losses.self_supervised_losses import TaylorH2VarianceModel, compute_self_supervised_loss

OUTPUT = ROOT / 'outputs/mean_refinement_round2_20260908'
EXPERIMENTS = ('r0_continue', 'r1_no_mean', 'r2_input_scale',
               'r3_mean_shape_005', 'r4_mean_shape_010')
TABLES = ('metrics', 'brightness', 'point_targets', 'point_pairs', 'lines',
          'depth_profiles', 'axial_points')
LABELS = {'baseline': 'Baseline', 'e3': 'Previous E3', 'r0_continue': 'R0 continue',
          'r1_no_mean': 'R1 no learned mean', 'r2_input_scale': 'R2 input scale',
          'r3_mean_shape_005': 'R3 mean gradient 5%',
          'r4_mean_shape_010': 'R4 mean gradient 10%'}


def _finite_mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) not in (None, '')]
    return float(np.mean(values)) if values else float('nan')


def _rate(rows, key):
    return float(np.mean([str(row[key]).lower() in ('true', '1') for row in rows])) if rows else float('nan')


def _method_parts(method):
    return method.rsplit('_', 1)


def _prediction_path(method, case_id):
    experiment, role = _method_parts(method)
    root = OLD_OUTPUT if experiment in ('baseline', 'e3') else OUTPUT
    return root / 'evaluation' / experiment / role / case_id / 'reconstruction.npy'


def _input_fingerprint(case):
    subset = case['path'] / 'subsets' / f"subset_{case['subset']:02d}.mat"
    # The driver verifies original raw-frame hashes before any evaluation. The
    # exact subset file binds frame indices, RL3 inputs and holdout statistics.
    return {'subset_path': str(subset), 'subset_sha256': sha256(subset),
            'prepared_sha256': sha256(case['path'] / 'prepared.mat')}


def _check_resume(marker, expected, paths):
    record = json.loads(marker.read_text())
    if not record.get('complete'):
        raise ValueError(f'Invalid completion record: {marker}')
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f'Resume fingerprint mismatch ({key}): {marker}')
    for key, path in paths.items():
        if not path.exists() or sha256(path) != record.get(key):
            raise ValueError(f'Resume artifact mismatch: {path}')
    return record


def _save_case(destination, prediction, anchor, record):
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / 'reconstruction.npy', prediction)
    np.save(destination / 'anchor.npy', anchor)
    record.update(complete=True,
                  prediction_sha256=sha256(destination / 'reconstruction.npy'),
                  anchor_sha256=sha256(destination / 'anchor.npy'))
    write_json(destination / 'complete.json', record)


def evaluate(experiment, resume=False):
    """Evaluate best and final using only the first visible GPU; never train."""
    if experiment not in EXPERIMENTS:
        raise ValueError(f'Unknown experiment {experiment!r}')
    from tools.mean_round2_experiment import build_model, forward, loss, load_operator
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    install_data()
    brightness_marker = OLD_OUTPUT / 'brightness_inputs/complete.json'
    if not brightness_marker.exists():
        raise ValueError('The previously verified MATLAB brightness inputs are missing')
    brightness_manifest = json.loads(brightness_marker.read_text())
    brightness_check = OLD_OUTPUT / 'brightness_inputs/verification.json'
    if not brightness_manifest.get('complete') or len(brightness_manifest.get('files', {})) != 12:
        raise ValueError('Incomplete original brightness input inventory')
    if not brightness_check.exists() or not json.loads(brightness_check.read_text()).get('passed'):
        raise ValueError('Original brightness statistics/RL3 verification did not pass')
    for relative, expected_sha in brightness_manifest['files'].items():
        if sha256(OLD_OUTPUT / relative) != expected_sha:
            raise ValueError(f'Original brightness input changed: {relative}')
    cases = all_cases()
    if len(cases) != 114:
        raise ValueError(f'Unexpected original case inventory: {len(cases)}')
    fingerprints = {case['id']: _input_fingerprint(case) for case in cases}
    _, geometry = load_config()
    result_dir = OUTPUT / 'evaluation' / experiment
    result_dir.mkdir(parents=True, exist_ok=True)
    tables = {name: [] for name in TABLES}
    operator = None
    for role, filename in (('best', 'checkpoint_best.pt'), ('final', 'checkpoint_last.pt')):
        checkpoint_path = OUTPUT / experiment / filename
        checkpoint_hash = sha256(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        config = checkpoint['config']
        if config.get('round2', {}).get('kind') != experiment:
            raise ValueError(f'Wrong experiment in checkpoint: {checkpoint_path}')
        if config.get('three_way', {}).get('kind') != 'e3':
            raise ValueError('Round-two checkpoints must retain the E3 base adapter')
        extra_step = int(checkpoint['completed_steps'])
        if role == 'final' and extra_step != 200:
            raise ValueError('Final checkpoint must contain 200 additional optimizer steps')
        method = f'{experiment}_{role}'
        model = build_model(config, initial=False).to(device)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()
        if operator is None:
            operator = load_operator(config, device)
        variance = TaylorH2VarianceModel(operator)
        for case in cases:
            destination = result_dir / role / case['id']
            marker = destination / 'complete.json'
            expected = {'checkpoint_sha256': checkpoint_hash,
                        'input_fingerprint': fingerprints[case['id']]}
            if resume and marker.exists():
                record = _check_resume(marker, expected, {
                    'prediction_sha256': destination / 'reconstruction.npy',
                    'anchor_sha256': destination / 'anchor.npy'})
                prediction = np.load(destination / 'reconstruction.npy')
                row = record['metrics']
                if case['split'] == 'priority':
                    truth = pa._read_yxz(case['path'] / 'prepared.mat', 'ground_truth')
            else:
                item = input_item(case['path'], case['subset'], device)
                with torch.inference_mode():
                    out, beta0 = forward(model, item, operator, config=config)
                    prediction = out.reconstruction[0, 0].cpu().numpy()
                    physical_anchor = getattr(out, '_physical_anchor', None)
                    if physical_anchor is None:
                        physical_anchor = item['f_var'] * out.beta[:, None, None, None, None]
                    anchor = physical_anchor[0, 0].cpu().numpy()
                    # Holdout statistics and GT are read only after prediction.
                    target = targets(case, device)
                    scoring = {**item, **target}
                    common = compute_self_supervised_loss(
                        out.reconstruction, target['measured_mean'], target['measured_variance'],
                        operator, variance, **config['loss'])
                    native = loss(out, scoring, operator, variance, config)
                    row = {'method': method, 'experiment': experiment, 'checkpoint_role': role,
                           'weight_step': 200 + extra_step, 'additional_weight_steps': extra_step,
                           'sample_id': case['sample'], 'case_id': case['id'],
                           'subset': case['subset'], 'split': case['split'],
                           'common_mean_loss': float(common.normalized_mean),
                           'common_variance_loss': float(common.normalized_var),
                           'native_total_loss': float(native.total),
                           'native_mean_loss': float(native.normalized_mean),
                           'native_variance_loss': float(native.normalized_var),
                           'native_weighted_tv': float(native.weighted_tv),
                           'output_max': float(prediction.max()), 'output_sum': float(prediction.sum()),
                           'beta0': float(beta0), 'beta': float(out.beta)}
                    if case['split'] != 'zero':
                        truth = target['ground_truth'][0, 0].cpu().numpy()
                        row.update(pa._score_structure_metrics(prediction, truth))
                        from scipy.ndimage import binary_dilation
                        support = binary_dilation(truth.max(0) > .1 * max(float(truth.max()), 1e-30), iterations=2)
                        row['background_xy_mass_fraction'] = float(prediction[:, ~support].sum() / max(float(prediction.sum()), 1e-30))
                    if case['id'] == 'P09_subset_01':
                        contaminated = dict(item)
                        contaminated['ground_truth'] = torch.ones_like(out.reconstruction) * 999
                        contaminated['measured_mean'] = torch.ones_like(item['input_mean']) * 999
                        contaminated['measured_variance'] = torch.ones_like(item['input_mean']) * 999
                        repeated, _ = forward(model, item, operator, config=config)
                        checked, _ = forward(model, contaminated, operator, config=config)
                        norm = out.reconstruction.norm().clamp_min(1e-30)
                        repeat_l2 = float((repeated.reconstruction - out.reconstruction).norm() / norm)
                        leak_l2 = float((checked.reconstruction - out.reconstruction).norm() / norm)
                        if max(repeat_l2, leak_l2) > 1e-6:
                            raise ValueError('Input repeatability / GT and holdout contamination check failed')
                        row.update(same_input_repeat_relative_l2=repeat_l2,
                                   gt_target_contamination_relative_l2=leak_l2,
                                   gt_target_contamination_max_change=float((checked.reconstruction - out.reconstruction).abs().max()))
                    record = {**expected, 'checkpoint_role': role, 'weight_step': 200 + extra_step,
                              'additional_weight_steps': extra_step, 'inference_target_or_gt_used': False,
                              'input_indices': np.asarray(item['input_indices']).tolist(), 'metrics': row,
                              'original_acquisition_hash_audit': str(OLD_OUTPUT / 'preflight.json')}
                    _save_case(destination, prediction, anchor, record)
            tables['metrics'].append(row)
            if case['split'] == 'priority':
                methods = {method: prediction}
                if case['family'] == 'points':
                    points, pairs, profiles = pa._point_metrics(geometry, case['scene_id'], case['repeat'], truth, methods)
                    tables['point_targets'].extend(points)
                    tables['point_pairs'].extend(pairs)
                    tables['depth_profiles'].extend(profiles)
                elif case['family'] == 'lines':
                    tables['lines'].extend(pa._line_metrics(geometry, case['scene_id'], case['repeat'], methods))
                else:
                    axial, pairs = pa._axial_metrics(geometry, case['repeat'], methods)
                    tables['axial_points'].extend(axial)
                    tables['point_pairs'].extend(pairs)
            print(json.dumps({'evaluated': method, 'case': case['id'], 'weight_step': 200 + extra_step}), flush=True)
        for sample, directory in (('points_z060_r01', PRIORITY / 'generated/points_z060_r01'),
                                  ('lines_z060_r01', PRIORITY / 'generated/lines_z060_r01'),
                                  ('P09', DATA / 'P09')):
            base = input_item(directory, 1, device)
            base_case = {'sample': sample, 'subset': 1, 'split': 'brightness', 'path': directory}
            reference = None
            target = None
            for gain in (1., 0., .1, .5, 2.):
                destination = result_dir / role / 'brightness' / sample
                destination.mkdir(parents=True, exist_ok=True)
                pred_path = destination / f'gain_{gain:g}.npy'
                marker = destination / f'gain_{gain:g}.json'
                source_path = OLD_OUTPUT / 'brightness_inputs' / sample / f'gain_{gain:g}.mat' if gain else directory / 'subsets/subset_01.mat'
                expected = {'checkpoint_sha256': checkpoint_hash, 'input_sha256': sha256(source_path),
                            'gain': gain, 'synthetic_exact_zero': gain == 0}
                if resume and marker.exists():
                    saved = _check_resume(marker, expected, {'prediction_sha256': pred_path})
                    pred = torch.from_numpy(np.load(pred_path)).to(device)[None, None]
                    row = saved['metrics']
                else:
                    item = brightness_item(sample, gain, base, device)
                    with torch.inference_mode():
                        out, _ = forward(model, item, operator, config=config)
                        pred = out.reconstruction
                        # Even this diagnostic loads targets after inference.
                        if target is None:
                            target = targets(base_case, device)
                        common = compute_self_supervised_loss(
                            pred, target['measured_mean'] * gain, target['measured_variance'] * gain ** 2,
                            operator, variance, **config['loss'])
                        if gain == 1.:
                            reference = pred.clone()
                        error = float((pred - reference * gain).norm() / (reference * gain).norm().clamp_min(1e-30)) if gain else None
                        row = {'method': method, 'sample_id': sample, 'gain': gain,
                               'output_max': float(pred.max()), 'output_sum': float(pred.sum()),
                               'relative_scale_error': error, 'mean_loss': float(common.normalized_mean),
                               'variance_loss': float(common.normalized_var),
                               'raw_frames_scaled_and_RL3_rerun': gain != 0}
                        np.save(pred_path, pred[0, 0].cpu().numpy())
                        write_json(marker, {**expected, 'complete': True, 'prediction_sha256': sha256(pred_path),
                                            'inference_target_or_gt_used': False, 'metrics': row})
                if gain == 1.:
                    reference = pred.clone()
                tables['brightness'].append(row)
        del model
        torch.cuda.empty_cache()
    for name, values in tables.items():
        csv_write(result_dir / f'{name}.csv', values)
    write_json(result_dir / 'complete.json', {'complete': True, 'experiment': experiment,
               'cases_per_checkpoint': len(cases), 'checkpoints': ['best', 'final'],
               'brightness_cases': len(tables['brightness']), 'actual_weight_steps_used': True,
               'table_sha256': {name: sha256(result_dir / f'{name}.csv') for name in TABLES},
               'precision': {'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                             'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32},
               'evaluation_source_sha256': sha256(Path(__file__))})


def _summary(tables, methods):
    rows = []
    for method in methods:
        metrics = [r for r in tables['metrics'] if r['method'] == method]
        test = [r for r in metrics if r['split'] == 'test']
        validation = [r for r in metrics if r['split'] == 'validation']
        points = [r for r in tables['point_targets'] if r['method'] == method and float(r['threshold']) == .1]
        pairs = [r for r in tables['point_pairs'] if r['method'] == method and float(r['threshold']) == .1]
        lines = [r for r in tables['lines'] if r['method'] == method and float(r['threshold']) == .1]
        zero = next(r for r in metrics if r['split'] == 'zero')
        brightness = [r for r in tables['brightness'] if r['method'] == method and float(r['gain']) in (.1, .5, 2.)]
        row = {'method': method, 'weight_step': int(metrics[0]['weight_step']),
               'test_aligned_nrmse': _finite_mean(test, 'gt_scale_aligned_nrmse'),
               'test_axial_w1_um': _finite_mean(test, 'gt_axial_w1_um'),
               'test_local_axial_w1_um': _finite_mean(test, 'local_axial_w1_um'),
               'test_background_xy_mass_fraction': _finite_mean(test, 'background_xy_mass_fraction'),
               'test_mean_loss': _finite_mean(test, 'common_mean_loss'),
               'test_variance_loss': _finite_mean(test, 'common_variance_loss'),
               'validation_aligned_nrmse': _finite_mean(validation, 'gt_scale_aligned_nrmse'),
               'single_localization_rate': _rate([r for r in points if r['kind'] == 'single'], 'matched'),
               'local_depth_w1_um': _finite_mean(points, 'local_depth_w1_um'),
               'lateral_pair_separation_rate': _rate([r for r in pairs if r['scene_id'].startswith('points')], 'separated'),
               'axial_pair_separation_rate': _rate([r for r in pairs if r['scene_id'] == 'axial_pairs'], 'separated'),
               'continuous_false_gap_rate': _rate([r for r in lines if float(r['gap_um']) == 0], 'false_gap'),
               'broken_bridge_rate': _rate([r for r in lines if float(r['gap_um']) > 0], 'bridged'),
               'gap4_bridge_rate': _rate([r for r in lines if float(r['gap_um']) == 4], 'bridged'),
               'weak_pair8_separation_rate': _rate([r for r in pairs if r['axis'] != 'z' and float(r['separation_um']) == 8 and float(r['ratio']) == .5], 'separated'),
               'zero_max': float(zero['output_max']), 'zero_sum': float(zero['output_sum']),
               'brightness_relative_scale_error': _finite_mean(brightness, 'relative_scale_error')}
        rows.append(row)
    return rows


def _detailed_tables(tables, methods, analysis):
    object_rows, repeat_rows, line_rows, pair_rows = [], [], [], []
    for method in methods:
        metrics = [r for r in tables['metrics'] if r['method'] == method]
        for split, samples in (('test', ('P07', 'T01', 'T02')), ('validation', ('P09', 'V01', 'V02'))):
            for sample in samples:
                selected = [r for r in metrics if r['split'] == split and r['sample_id'] == sample]
                object_rows.append({'method': method, 'sample_id': sample, 'split': split,
                    **{key: _finite_mean(selected, key) for key in (
                        'gt_scale_aligned_nrmse', 'gt_axial_w1_um', 'local_axial_w1_um',
                        'common_mean_loss', 'common_variance_loss', 'background_xy_mass_fraction')}})
        for threshold in (.05, .1, .2):
            points = [r for r in tables['point_targets'] if r['method'] == method and float(r['threshold']) == threshold]
            pairs = [r for r in tables['point_pairs'] if r['method'] == method and float(r['threshold']) == threshold]
            lines = [r for r in tables['lines'] if r['method'] == method and float(r['threshold']) == threshold]
            for repeat in (1, 2, 3):
                p = [r for r in points if int(r['repeat']) == repeat]
                pp = [r for r in pairs if int(r['repeat']) == repeat]
                ln = [r for r in lines if int(r['repeat']) == repeat]
                repeat_rows.append({'method': method, 'threshold': threshold, 'repeat': repeat,
                    'single_localization_rate': _rate([r for r in p if r['kind'] == 'single'], 'matched'),
                    'local_depth_w1_um': _finite_mean(p, 'local_depth_w1_um'),
                    'lateral_pair_separation_rate': _rate([r for r in pp if r['scene_id'].startswith('points')], 'separated'),
                    'axial_pair_separation_rate': _rate([r for r in pp if r['scene_id'] == 'axial_pairs'], 'separated'),
                    'false_gap_rate': _rate([r for r in ln if float(r['gap_um']) == 0], 'false_gap'),
                    'gap4_bridge_rate': _rate([r for r in ln if float(r['gap_um']) == 4], 'bridged'),
                    'weak_pair8_separation_rate': _rate([r for r in pp if r['axis'] != 'z' and float(r['separation_um']) == 8 and float(r['ratio']) == .5], 'separated')})
            for gap in (0, 4, 8, 12):
                for angle in (0, 45, 90):
                    for amplitude in (1., .5):
                        selected = [r for r in lines if float(r['gap_um']) == gap and float(r['angle_deg']) == angle and float(r['amplitude']) == amplitude]
                        line_rows.append({'method': method, 'threshold': threshold, 'gap_um': gap,
                            'angle_deg': angle, 'amplitude': amplitude, 'n': len(selected),
                            'false_gap_rate': _rate(selected, 'false_gap'), 'bridge_rate': _rate(selected, 'bridged'),
                            'endpoint_retention': _finite_mean(selected, 'endpoint_retention_fraction'),
                            'coverage': _finite_mean(selected, 'coverage_fraction')})
            groups = sorted({(float(r['separation_um']), r['axis'], float(r['ratio'])) for r in pairs})
            for separation, axis, ratio in groups:
                selected = [r for r in pairs if float(r['separation_um']) == separation and r['axis'] == axis and float(r['ratio']) == ratio]
                pair_rows.append({'method': method, 'threshold': threshold, 'separation_um': separation,
                    'axis': axis, 'ratio': ratio, 'n': len(selected),
                    'both_localized_rate': _rate(selected, 'both_localized'), 'separation_rate': _rate(selected, 'separated')})
    csv_write(analysis / 'per_object_summary.csv', object_rows)
    csv_write(analysis / 'per_repeat_summary.csv', repeat_rows)
    csv_write(analysis / 'line_quality_summary.csv', line_rows)
    csv_write(analysis / 'pair_resolution_summary.csv', pair_rows)
    lookup = {(r['method'], r['case_id']): r for r in tables['metrics'] if r['split'] in ('test', 'priority')}
    failures = []
    for (method, case_id), row in lookup.items():
        experiment, role = _method_parts(method)
        if experiment not in EXPERIMENTS:
            continue
        for reference in dict.fromkeys((f'r0_continue_{role}', f'baseline_{role}', f'e3_{role}', 'e3_final')):
            if reference == method:
                continue
            baseline = lookup[reference, case_id]
            failures.append({'method': method, 'reference_method': reference, 'case_id': case_id,
                             'split': row['split'],
                'shape_error_increase': float(row['gt_scale_aligned_nrmse']) - float(baseline['gt_scale_aligned_nrmse']),
                'local_depth_error_increase_um': float(row['local_axial_w1_um']) - float(baseline['local_axial_w1_um'])})
    failures.sort(key=lambda r: (r['reference_method'].startswith('r0_continue_'), r['shape_error_increase']), reverse=True)
    csv_write(analysis / 'failure_cases.csv', failures)
    return object_rows, repeat_rows, failures


def _figures(analysis, summary, tables, failures):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = analysis / 'actual_reconstruction_comparison'
    folder.mkdir(parents=True, exist_ok=True)
    sigma = float(np.load(OLD_OUTPUT / 'illumination/calibration_moments.npz')['sigma'])
    _, geometry = load_config()
    object_pixel_pitch_um = float(geometry['acquisition']['object_pixel_pitch_um'])
    inventory = {c['id']: c for c in all_cases()}
    case_ids = list(dict.fromkeys(['P07_subset_01', 'T01_subset_01', 'T02_subset_01',
             'points_z060_r01', 'lines_z060_r01', 'axial_pairs_r01'] +
             [r['case_id'] for r in failures[:3]]))
    entries = []
    profile_rows = []
    for case_id in case_ids:
        truth = pa._read_yxz(inventory[case_id]['path'] / 'prepared.mat', 'ground_truth')
        fixed_y, fixed_x = np.unravel_index(int(np.argmax(truth.sum(0))), truth.shape[1:])
        for role in ('best', 'final'):
            methods = ['baseline_final', 'e3_final', *[f'{e}_{role}' for e in EXPERIMENTS]]
            volumes = [truth * sigma, *[np.load(_prediction_path(m, case_id)) for m in methods]]
            labels = ['GT (fixed physical units)', 'Baseline final', 'Previous E3 final',
                      *[f'{LABELS[e]} {role}' for e in EXPERIMENTS]]
            for display in ('shape', 'shared'):
                vmax = 1. if display == 'shape' else max(float(v.max()) for v in volumes)
                fig, axes = plt.subplots(2, len(volumes), figsize=(25, 6))
                for col, (volume, label) in enumerate(zip(volumes, labels)):
                    shown = volume / max(float(volume.max()), 1e-30) if display == 'shape' else volume
                    axes[0, col].imshow(shown.max(0), cmap='magma', vmin=0, vmax=vmax)
                    axes[0, col].set_title(label, fontsize=9)
                    axes[0, col].axis('off')
                    axes[1, col].imshow(shown.max(1), cmap='magma', vmin=0, vmax=vmax,
                                        aspect='auto', extent=[0, object_pixel_pitch_um * volume.shape[-1], 105, 5])
                    axes[1, col].set_xlabel('X, um')
                    axes[1, col].set_ylabel('Z, um')
                suffix = 'one normalization per entire volume; never per layer' if display == 'shape' else 'one shared intensity scale; GT conversion fixed across all methods'
                fig.suptitle(f'{case_id} | {role} | {suffix}', fontsize=11)
                fig.tight_layout()
                name = f'{case_id}_{role}_{display}.png'
                fig.savefig(folder / name, dpi=150)
                plt.close(fig)
                entries.append({'case_id': case_id, 'role': role, 'display': display,
                                'path': str((folder / name).relative_to(OUTPUT)), 'methods': methods})
            for name, volume in zip(['GT', *methods], volumes):
                profile = volume[:, max(fixed_y - 2, 0):fixed_y + 3, max(fixed_x - 2, 0):fixed_x + 3].sum((1, 2))
                normalized = profile / max(float(profile.sum()), 1e-30)
                for z, raw, value in zip(range(10, 101, 10), profile, normalized):
                    profile_rows.append({'case_id': case_id, 'role_panel': role, 'method': name,
                        'fixed_x_pixel': int(fixed_x), 'fixed_y_pixel': int(fixed_y), 'z_um': z,
                        'raw_local_mass': float(raw), 'normalized_local_mass': float(value)})
    csv_write(analysis / 'representative_fixed_profiles.csv', profile_rows)
    atlas = ['# 第二轮实际重建对比', '',
             '每图从左到右：GT、原 baseline final、上一轮 E3 final、本轮 R0–R4。上排 XY 最大投影，下排 XZ 最大投影。',
             'best 和 final 分开显示。shape 使用每个完整三维体一个亮度比例；shared 使用全图共同亮度范围。原始预测没有重新定标。', '']
    for case_id in case_ids:
        atlas += [f'## {case_id}', '']
        for entry in [x for x in entries if x['case_id'] == case_id]:
            name = Path(entry['path']).name
            atlas += [f"{entry['role']} / {entry['display']}", '', f'![{name}]({name})', '']
    (folder / 'README_ZH.md').write_text('\n'.join(atlas))
    methods = [r['method'] for r in summary]
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for ax, key, title in zip(axes.ravel(),
        ('test_aligned_nrmse', 'test_local_axial_w1_um', 'gap4_bridge_rate',
         'continuous_false_gap_rate', 'weak_pair8_separation_rate', 'test_background_xy_mass_fraction'),
        ('Test shape error (lower)', 'Test local depth W1, um (lower)', '4 um gap false bridges (lower)',
         'Continuous line false breaks (lower)', '8 um weak pair separation (higher)', 'Background mass fraction (lower)')):
        ax.bar(methods, [r[key] for r in summary])
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=80, labelsize=8)
    fig.tight_layout()
    fig.savefig(analysis / 'comparison.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 7, figsize=(27, 8), sharex=True, sharey=True)
    for ax, method in zip(axes.ravel(), methods):
        for repeat in (1, 2, 3):
            for cell in ('1', '2', '3', '4'):
                rows = [r for r in tables['point_targets'] if r['method'] == method and r['kind'] == 'single'
                        and int(r['repeat']) == repeat and str(r['cell_id']) == cell and float(r['threshold']) == .1]
                rows.sort(key=lambda r: float(r['z_um']))
                ax.plot([float(r['z_um']) for r in rows],
                        [float(r.get('pred_z_layer') or 'nan') * 10 + 10 for r in rows], alpha=.55)
        ax.plot([10, 100], [10, 100], 'k--')
        ax.set_title(method, fontsize=9)
        ax.set_xlabel('True depth, um')
        ax.set_ylabel('Predicted depth, um')
    fig.tight_layout()
    fig.savefig(analysis / 'depth_following_all_repeats.png', dpi=150)
    plt.close(fig)
    return entries



def _training_diagnostics(analysis):
    """Aggregate per-sample q diagnostics, excluding warmup/evaluation records."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summaries, by_step, sources = [], [], {}
    figure, axes = plt.subplots(4, len(EXPERIMENTS), figsize=(25, 14), squeeze=False)
    for column, experiment in enumerate(EXPERIMENTS):
        directory = OUTPUT / experiment
        records = []
        paths = sorted(directory.glob('gradient_diagnostics_rank*.jsonl'))
        if not paths:
            raise ValueError(f'Missing per-sample gradient diagnostics: {experiment}')
        for path in paths:
            sources[str(path)] = sha256(path)
            for number, line in enumerate(path.read_text().splitlines(), 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f'Incomplete gradient record: {path}:{number}') from error
                if row.get('phase') == 'training':
                    records.append(row)
        checkpoints = {}
        for role, filename in (('best', 'checkpoint_best.pt'), ('final', 'checkpoint_last.pt')):
            path = directory / filename
            sources[str(path)] = sha256(path)
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            step = int(checkpoint['completed_steps'])
            gamma = float(checkpoint['model_state']['mean_gain_gamma'])
            bound = float(checkpoint['config']['three_way']['gain_bound'])
            checkpoints[role] = {'additional_step': step, 'weight_step': 200 + step,
                'gamma': gamma, 'mean_gain_multiplier': 1. + bound * math.tanh(gamma)}
            if role == 'final':
                expected_batch = int(checkpoint['config']['optimization']['global_batch_size'])
                denominator_floor = float(checkpoint['config'].get('round2', {}).get('shape_gradient_eps', 1e-12))
            del checkpoint
        if checkpoints['final']['additional_step'] != 200:
            raise ValueError(f'Gradient report requires the completed 200-step run: {experiment}')
        training_path = directory / 'training_metrics.csv'
        validation_path = directory / 'validation_metrics.csv'
        sources[str(training_path)] = sha256(training_path)
        sources[str(validation_path)] = sha256(validation_path)
        training_rows = csv_read(training_path)
        validation_rows = [r for r in csv_read(validation_path) if r.get('split') == 'validation']
        training = {int(r['step']): r for r in training_rows}
        if len(training) != len(training_rows):
            raise ValueError(f'Duplicate optimizer steps in training metrics: {experiment}')
        steps = sorted({int(r['update_step']) for r in records})
        if steps != list(range(1, 201)):
            raise ValueError(f'Missing or unexpected training gradient steps: {experiment}')
        for step in steps:
            selected = [r for r in records if int(r['update_step']) == step]
            if len(selected) != expected_batch:
                raise ValueError(f'Gradient count differs from global batch: {experiment} step {step}: {len(selected)} vs {expected_batch}')
            if step not in training:
                raise ValueError(f'Missing training metric step {step}: {experiment}')
            tv_ratios = [float(r['weighted_tv_q_gradient_norm']) / (float(r['var_q_gradient_norm']) + denominator_floor) for r in selected]
            item = {'experiment': experiment, 'phase': 'training', 'additional_step': step,
                    'weight_step': 200 + step, 'sample_count': len(selected),
                    'shape_coefficient_mean': _finite_mean(selected, 'shape_coefficient'),
                    'shape_coefficient_max': max(float(r['shape_coefficient']) for r in selected),
                    'shape_gradient_ratio_mean': _finite_mean(selected, 'shape_gradient_ratio'),
                    'shape_gradient_ratio_max': max(float(r['shape_gradient_ratio']) for r in selected),
                    'effective_gradient_budget': _finite_mean(selected, 'effective_gradient_budget'),
                    'mean_var_gradient_cosine_mean': _finite_mean(selected, 'mean_var_gradient_cosine'),
                    'direction_conflict_fraction': float(np.mean([float(r['mean_var_gradient_cosine']) < 0 for r in selected])),
                    'weighted_tv_to_var_q_gradient_mean': float(np.mean(tv_ratios)),
                    'weighted_tv_to_var_q_gradient_max': float(np.max(tv_ratios)),
                    'var_q_gradient_norm_mean': _finite_mean(selected, 'var_q_gradient_norm'),
                    'mean_shape_q_gradient_norm_mean': _finite_mean(selected, 'mean_shape_q_gradient_norm'),
                    'all_gradients_finite': all(bool(r['gradients_finite']) for r in selected),
                    'all_budgets_passed': all(bool(r['budget_passed']) for r in selected)}
            train = training[step]
            item['train_native_total_loss'] = float(train['total_loss'])
            item['train_shared_e3_loss'] = sum(float(train[k]) for k in ('normalized_mean_loss', 'normalized_var_loss', 'weighted_tv_loss'))
            validation_step = [r for r in validation_rows if int(r['step']) == step]
            samples = sorted({r['sample_id'] for r in validation_step})
            item['validation_shared_selection_score'] = float(np.mean([
                _finite_mean([r for r in validation_step if r['sample_id'] == sample], 'selection_score')
                for sample in samples])) if samples else None
            if not item['all_gradients_finite'] or not item['all_budgets_passed']:
                raise ValueError(f'Failed recorded gradient acceptance: {experiment} step {step}')
            by_step.append(item)
        selected_steps = [r for r in by_step if r['experiment'] == experiment]
        ratios = [float(r['shape_gradient_ratio']) for r in records]
        tv_ratios = [float(r['weighted_tv_q_gradient_norm']) / (float(r['var_q_gradient_norm']) + denominator_floor) for r in records]
        summary = {'experiment': experiment, 'phase': 'training', 'optimizer_steps': len(steps),
            'sample_count': len(records), 'expected_global_batch': expected_batch,
            'shape_coefficient_mean': _finite_mean(records, 'shape_coefficient'),
            'shape_coefficient_max': max(float(r['shape_coefficient']) for r in records),
            'shape_gradient_ratio_mean': float(np.mean(ratios)), 'shape_gradient_ratio_max': max(ratios),
            'shape_gradient_budget': _finite_mean(records, 'shape_gradient_budget'),
            'mean_var_gradient_cosine_mean': _finite_mean(records, 'mean_var_gradient_cosine'),
            'direction_conflict_fraction': float(np.mean([float(r['mean_var_gradient_cosine']) < 0 for r in records])),
            'weighted_tv_to_var_q_gradient_mean': float(np.mean(tv_ratios)),
            'weighted_tv_to_var_q_gradient_max': float(np.max(tv_ratios)),
            'all_gradients_finite': all(r['all_gradients_finite'] for r in selected_steps),
            'all_budgets_passed': all(r['all_budgets_passed'] for r in selected_steps)}
        for role, values in checkpoints.items():
            summary.update({f'{role}_{key}': value for key, value in values.items()})
        summaries.append(summary)
        x = [r['weight_step'] for r in selected_steps]
        axes[0, column].plot(x, [r['train_shared_e3_loss'] for r in selected_steps], alpha=.65, label='Train, shared E3 score')
        val = [r for r in selected_steps if r['validation_shared_selection_score'] is not None]
        axes[0, column].plot([r['weight_step'] for r in val], [r['validation_shared_selection_score'] for r in val], 'o-', label='Validation, shared E3 score')
        axes[0, column].axvline(checkpoints['best']['weight_step'], color='k', ls='--', alpha=.5, label='Best checkpoint')
        axes[0, column].set_title(LABELS[experiment])
        axes[0, column].set_ylabel('Shared score; lower better')
        axes[1, column].plot(x, [r['shape_coefficient_mean'] for r in selected_steps], label='Mean loss coefficient')
        axes[1, column].plot(x, [r['shape_gradient_ratio_mean'] for r in selected_steps], label='Actual q-gradient ratio')
        axes[1, column].plot(x, [r['effective_gradient_budget'] for r in selected_steps], 'k--', label='Per-sample budget')
        axes[1, column].set_ylabel('Coefficient / q-gradient ratio')
        axes[2, column].plot(x, [r['mean_var_gradient_cosine_mean'] for r in selected_steps], label='Mean cosine')
        axes[2, column].plot(x, [r['direction_conflict_fraction'] for r in selected_steps], label='Fraction cosine < 0')
        axes[2, column].axhline(0, color='k', lw=.5)
        axes[2, column].set_ylabel('Unweighted direction diagnostic')
        axes[3, column].semilogy(x, [max(r['weighted_tv_to_var_q_gradient_mean'], 1e-30) for r in selected_steps], label='Weighted TV / variance at q')
        axes[3, column].set_ylabel('Gradient norm ratio (log scale)')
        for row in range(4):
            axes[row, column].set_xlabel('Total optimizer steps')
            axes[row, column].legend(fontsize=7)
    figure.suptitle('Training diagnostics: individual samples first, then step averages; no gradient projection')
    figure.tight_layout()
    figure.savefig(analysis / 'training_validation_gradient_trends.png', dpi=150)
    plt.close(figure)
    csv_write(analysis / 'gradient_by_step.csv', by_step)
    csv_write(analysis / 'gradient_summary.csv', summaries)
    return summaries, sources


def report():
    """Aggregate the 14 frozen models and generate figures without GPU use."""
    analysis = OUTPUT / 'analysis'
    analysis.mkdir(parents=True, exist_ok=True)
    tables = {name: [] for name in TABLES}
    sources = {}
    for experiment in ('baseline', 'e3', *EXPERIMENTS):
        root = OLD_OUTPUT if experiment in ('baseline', 'e3') else OUTPUT
        folder = root / 'evaluation' / experiment
        if not (folder / 'complete.json').exists():
            raise ValueError(f'Incomplete evaluation: {experiment}')
        for name in TABLES:
            path = folder / f'{name}.csv'
            tables[name].extend(csv_read(path))
            sources[str(path)] = sha256(path)
    for name, values in tables.items():
        csv_write(analysis / f'{name}.csv', values)
    methods = [f'{e}_{r}' for e in ('baseline', 'e3', *EXPERIMENTS) for r in ('best', 'final')]
    summary = _summary(tables, methods)
    csv_write(analysis / 'summary.csv', summary)
    write_json(analysis / 'summary.json', summary)
    objects, repeats, failures = _detailed_tables(tables, methods, analysis)
    figure_entries = _figures(analysis, summary, tables, failures)
    write_json(analysis / 'figure_manifest.json', figure_entries)
    gradient_summary, gradient_sources = _training_diagnostics(analysis)
    sources.update(gradient_sources)
    lookup = {r['method']: r for r in summary}
    engineering = []
    for experiment in EXPERIMENTS[1:]:
        for role in ('best', 'final'):
            row, ref = lookup[f'{experiment}_{role}'], lookup[f'r0_continue_{role}']
            deltas = {key: row[key] - ref[key] for key in (
                'gap4_bridge_rate', 'weak_pair8_separation_rate', 'continuous_false_gap_rate',
                'test_aligned_nrmse', 'test_local_axial_w1_um')}
            checks = {'gap4_reduced_at_least_10pp': deltas['gap4_bridge_rate'] <= -.1 + 1e-12,
                      'weak_pair8_drop_at_most_5pp': deltas['weak_pair8_separation_rate'] >= -.05 - 1e-12,
                      'continuous_false_gap_increase_at_most_5pp': deltas['continuous_false_gap_rate'] <= .05 + 1e-12,
                      'test_shape_not_worse': deltas['test_aligned_nrmse'] <= 1e-12,
                      'test_local_depth_increase_at_most_1um': deltas['test_local_axial_w1_um'] <= 1. + 1e-12}
            engineering.append({'method': row['method'], 'reference_method': ref['method'],
                                **{f'delta_{key}': value for key, value in deltas.items()}, **checks,
                                'passes_all': all(checks.values()), 'used_for_checkpoint_selection': False})
    csv_write(analysis / 'engineering_acceptance.csv', engineering)
    text = ['# 均值与方差结合：第二轮结果', '',
        '本轮五组从同一个上一轮 E3 final（200步）开始，额外训练200步。显示的总步数为200加本轮步数。所有组起始 gamma 归零，网络学习率为1e-4，beta/gamma学习率为1e-5。',
        '本轮五组训练及推理统一关闭TF32，使用完整FP32。旧baseline/E3直接复用上一轮冻结预测及原运行条件；新优化优先与同精度、同额外训练量的R0比较。',
        'R0 继续原 E3；R1 关闭可学习的均值亮度项、保留输入均值解析定标；R2 增加十帧输入尺度归一化；R3/R4 在结构分支增加受限的均值形状梯度。',
        '**5%和10%是结构 q 处的梯度预算，经过前50步逐渐启用；不是 mean loss 固定权重，也不是 Adam 参数更新比例。** 验证和 best 选择使用统一的旧 E3 目标，不含新增结构梯度项。',
        '照明NA沿用0.05，检测PSF沿用0.15。没有新增GT训练，也没有使用测试对象或局部结构图挑选checkpoint。原测试图已在前轮诊断中查看，本报告不把它们称为从未见过的新测试集。', '',
        '| 方法 | 总步数 | 形状误差↓ | 局部深度误差µm↓ | 背景占比↓ | 4µm断口误连↓ | 8µm弱点对分开↑ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        text.append(f"| {row['method']} | {row['weight_step']} | {row['test_aligned_nrmse']:.4f} | {row['test_local_axial_w1_um']:.2f} | {row['test_background_xy_mass_fraction']:.2%} | {row['gap4_bridge_rate']:.1%} | {row['weak_pair8_separation_rate']:.1%} |")
    text += ['', '下面逐组比较各自 final 与继续训练对照 R0 final。负的误差差值表示改善，不能只据均值误差下降判断结构变好。', '',
             '| 实验 | 形状误差变化↓ | 局部深度变化µm↓ | 4µm误连变化↓ | 连续线假断变化↓ |',
             '|---|---:|---:|---:|---:|']
    control = lookup['r0_continue_final']
    for experiment in EXPERIMENTS[1:]:
        row = lookup[f'{experiment}_final']
        text.append(f"| {experiment} | {row['test_aligned_nrmse']-control['test_aligned_nrmse']:+.4f} | {row['test_local_axial_w1_um']-control['test_local_axial_w1_um']:+.2f} | {row['gap4_bridge_rate']-control['gap4_bridge_rate']:+.1%} | {row['continuous_false_gap_rate']-control['continuous_false_gap_rate']:+.1%} |")
    text += ['', '预先约定的工程筛查：相对同角色R0，10%阈值下4µm误连至少下降10个百分点，8µm弱点对分开率最多下降5个百分点，连续线假断最多增加5个百分点，测试平均形状误差不增加，测试平均局部深度误差最多增加1µm。它只用于冻结训练后的评价，不用于选择或更换checkpoint。', '',
             '| 方法 | 全部满足 | 未满足项 |', '|---|---|---|']
    criterion_labels = {'gap4_reduced_at_least_10pp': '4µm误连下降不足10个百分点',
                        'weak_pair8_drop_at_most_5pp': '8µm弱点对分开率下降超过5个百分点',
                        'continuous_false_gap_increase_at_most_5pp': '连续线假断增加超过5个百分点',
                        'test_shape_not_worse': '平均形状误差增加',
                        'test_local_depth_increase_at_most_1um': '平均局部深度误差增加超过1µm'}
    for row in engineering:
        failed = '；'.join(label for key, label in criterion_labels.items() if not row[key]) or '无'
        text.append(f"| {row['method']} | {'是' if row['passes_all'] else '否'} | {failed} |")
    if not any(row['passes_all'] for row in engineering):
        text += ['', '**本轮没有方法同时达到以上全部条件，不能宣布综合获胜。**']
    text += ['', '逐对象检查，避免一个对象退步被平均值遮住：', '',
             '| 方法 | 对象 | 形状误差↓ | 局部深度误差µm↓ |', '|---|---|---:|---:|']
    for row in objects:
        if row['split'] == 'test':
            text.append(f"| {row['method']} | {row['sample_id']} | {row['gt_scale_aligned_nrmse']:.4f} | {row['local_axial_w1_um']:.2f} |")
    text += ['', '阈值敏感性：每个格按 5% / 10% / 20% 检测阈值排列。三次独立采集分别保存在 per_repeat_summary.csv；此处只汇总指标，没有平均三次重建图。', '',
             '| 方法 | 连续线假断↓ | 8µm弱点对分开↑ |', '|---|---|---|']
    for method in methods:
        strings = []
        for key in ('false_gap_rate', 'weak_pair8_separation_rate'):
            strings.append(' / '.join(f"{_finite_mean([r for r in repeats if r['method']==method and r['threshold']==threshold],key):.1%}" for threshold in (.05, .1, .2)))
        text.append(f'| {method} | {strings[0]} | {strings[1]} |')
    text += ['', '训练过程中，mean形状项实际推了多大：', '',
             '| 实验 | 平均q系数 | 平均/最大q梯度比 | mean/var方向相反比例 | TV/var梯度比均值 | best/final亮度乘数 |',
             '|---|---:|---:|---:|---:|---:|']
    for row in gradient_summary:
        text.append(f"| {row['experiment']} | {row['shape_coefficient_mean']:.5g} | {row['shape_gradient_ratio_mean']:.3%} / {row['shape_gradient_ratio_max']:.3%} | {row['direction_conflict_fraction']:.1%} | {row['weighted_tv_to_var_q_gradient_mean']:.3g} | {row['best_mean_gain_multiplier']:.7f} / {row['final_mean_gain_multiplier']:.7f} |")
    text += ['', '以上仅统计phase=training，先计算每个样本的q处梯度，再按优化步和全程汇总。5%/10%限制每个样本额外mean形状梯度相对方差梯度的大小，前50步渐增；不是固定loss权重，不是模型参数或Adam更新的比例。余弦小于零表示两个原始方向相反，这里只记录诊断，没有执行梯度投影；R0/R1/R2也计算这个潜在方向，但其mean形状系数为零。',
             'TV比较的是加权TV与方差在q处的梯度范数，不能只看TV loss数值大小。训练趋势中的shared E3 score与验证口径一致；各组实际优化的native total另存gradient_by_step.csv。gamma、亮度乘数、系数、实际比例及预算检查完整记录在gradient_summary.csv。',
             '![训练、验证与梯度趋势](analysis/training_validation_gradient_trends.png)', '',
             '当前数据能回答到哪一步：', '']
    for experiment in EXPERIMENTS[1:]:
        row = lookup[f'{experiment}_final']
        improved = row['test_aligned_nrmse'] < control['test_aligned_nrmse'] and row['test_local_axial_w1_um'] < control['test_local_axial_w1_um']
        gap_safe = row['gap4_bridge_rate'] <= control['gap4_bridge_rate']
        text.append(f"- {experiment}：相对R0，形状和局部深度是否同时改善：{'是' if improved else '否'}；4µm误连是否未增加：{'是' if gap_safe else '否'}。还需对照逐对象和各次采集，不能据单个均值认定全面优于对照。")
    if all(row['axial_pair_separation_rate'] == 0 for row in summary):
        text.append('- 所有方法仍未分开本实验20–40µm轴向点对，不能声称轴向分辨能力已经突破。')
    text += ['', 'R0/R1用于判断可学习mean亮度项本身的增益；R3/R4与R0相比才用于检查受限mean结构梯度。R2是独立尺度改动，不能把R2收益算成mean结构梯度收益。',
             'R0/R1均从已有E3权重继续训练且gamma重置，因此该消融只能说明这段继续训练中可学习亮度项的作用，不能推出从随机初始化训练时mean loss完全无效。',
             '旧E3的mean loss原先只更新一个接近零的gamma，上一轮显著改善不能直接归因于mean梯度改善结构。所有共同物理分数使用同一旧H²前向；它的相关项缺失仍存在，评分只是诊断。',
             'best与final全部保留。对旧baseline的比较同时包含更多训练步数，优化本身应首先与本轮R0同计算量对照比较。', '',
             '[实际重建图册](analysis/actual_reconstruction_comparison/README_ZH.md)：P07、T01、T02及点线图，best/final分开，整卷归一化图和统一亮度图均提供。',
             '![统一指标](analysis/comparison.png)', '![三次深度跟随](analysis/depth_following_all_repeats.png)', '',
             '逐项数据：analysis/metrics.csv、per_object_summary.csv、per_repeat_summary.csv、line_quality_summary.csv、pair_resolution_summary.csv、failure_cases.csv。原始预测和anchor在evaluation/<实验>/<best或final>/<场景>/；每个结果附checkpoint、输入和产物SHA256。',
             '三个测试对象的十个重叠子集不当作十次独立采集；局部测试的三次独立照明重复逐次报告。']
    (OUTPUT / 'REPORT_ZH.md').write_text('\n'.join(text) + '\n')
    write_json(analysis / 'source_sha256.json', sources)
    write_json(OUTPUT / 'evaluation_complete.json', {'complete': True, 'experiments': list(EXPERIMENTS),
               'reference_methods': ['baseline_best', 'baseline_final', 'e3_best', 'e3_final'],
               'evaluated_methods': methods, 'summary': summary, 'finished_unix': time.time()})
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=EXPERIMENTS)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    if args.report:
        report()
    elif args.experiment:
        evaluate(args.experiment, resume=args.resume)
    else:
        parser.error('Choose --experiment or --report')
