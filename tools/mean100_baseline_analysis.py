"""Frozen final800 four-method analysis; best740 is supplementary, never primary.

No training or RL rerun. Historical evaluation definitions are reused read-only.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from collections import defaultdict
import numpy as np
import torch
from tools import v3_compare_evaluation as ev
from tools import v3_compare_local_audit as local
from tools import v5_mixed_real_evaluation as v5
from tools import v5_mixed_real_anchor_experiment as exp
from tools.priority_validation_common import load_config
from tools.three_way_checks import input_item
from losses.self_supervised_losses import TaylorH2VarianceModel
from utils.io import save_volume_tiff

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ('mean_rl3', 'taylor_rl3_sqrt', 'taylor100_final800', 'mean100_final800')
METHODS = PRIMARY + ('mean100_best740',)
LABELS = dict(zip(METHODS, ('Mean-RL3', 'Taylor-RL3-sqrt', 'Taylor100 final800',
                           'Mean100 final800', 'Mean100 best740')))
SHA = ev.old.sha256
write_csv = ev.csv_write


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def setup(root):
    registry = json.loads((ROOT / 'CURRENT_BASELINE.json').read_text())
    assert registry['baseline_id'] == 'mean100_v5_mean_anchor_mixed_real_no_p12_800_20260913'
    ev.exp.DATA = Path(registry['dataset_root'])
    cases = ev.all_cases()
    if (root / 'manifest.json').exists():
        manifest = json.loads((root / 'manifest.json').read_text())
        assert manifest['baseline_id'] == registry['baseline_id']
        return manifest, cases
    root.mkdir(parents=True, exist_ok=False)
    checkpoints = {}
    for method in METHODS[2:]:
        anchor = 'taylor' if method.startswith('taylor') else 'mean'
        filename = 'checkpoint_best.pt' if method.endswith('740') else 'checkpoint_step_000800.pt'
        path = ROOT / f'outputs/v5_sim_real_no_p12_{anchor}_anchor_e3_mean100_800_20260911_run01/{anchor}_anchor_e3_mean100' / filename
        payload = torch.load(path, map_location='cpu', weights_only=False)
        cfg = payload['config']
        step = 740 if method.endswith('740') else 800
        assert payload['completed_steps'] == step
        assert cfg['model']['reconstruction_anchor'] == ('mean_rl3' if anchor == 'mean' else 'taylor_sqrt')
        assert payload['dataset_fingerprint'] == registry['dataset_fingerprint']
        checkpoints[method] = dict(path=str(path), sha256=SHA(path), step=step,
                                   anchor=cfg['model']['reconstruction_anchor'], config=cfg)
        if method == 'mean100_final800':
            assert checkpoints[method]['sha256'] == registry['checkpoints']['final']['sha256']
        if method == 'taylor100_final800':
            best = torch.load(path.parent / 'checkpoint_best.pt', map_location='cpu', weights_only=False)
            assert best['completed_steps'] == 800
            assert all(torch.equal(v, best['model_state'][k]) for k, v in payload['model_state'].items())
    file_hashes = {}
    def digest(path):
        key = str(path.resolve())
        if key not in file_hashes:
            file_hashes[key] = SHA(path)
        return file_hashes[key]
    fingerprints = {}
    for c in cases:
        subset = c['path'] / 'subsets' / f"subset_{c['subset']:02d}.mat"
        fingerprints[c['id']] = {'subset_sha256': digest(subset), 'prepared_sha256': digest(c['path'] / 'prepared.mat')}
    manifest = dict(baseline_id=registry['baseline_id'], primary_methods=PRIMARY,
                    supplementary=['mean100_best740'], checkpoints=checkpoints,
                    taylor_best_model_state_identical_to_final800=True,
                    dataset_root=registry['dataset_root'], input_fingerprints=fingerprints,
                    cases=cases, cases_per_method=94, input_frames=10, rl_iterations=3,
                    real_reuse=registry['evaluation_root'], real_grouped_figures=registry['grouped_comparison'],
                    registry=registry, source_sha256=SHA(Path(__file__)))
    write_json(root / 'manifest.json', manifest)
    shutil.copy2(Path(__file__), root / 'evaluation_source_snapshot.py')
    shutil.copy2(ROOT / 'CURRENT_BASELINE.json', root / 'baseline_registry_snapshot.json')
    return manifest, cases


def evaluate(root):
    manifest, cases = setup(root)
    torch.set_num_threads(2)
    exp.configure_precision()
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    operator = v5.simulation_operator(device)
    models = {}
    for method, rec in manifest['checkpoints'].items():
        assert SHA(Path(rec['path'])) == rec['sha256']
        payload, cfg, model = v5.load_model(Path(rec['path']), device)
        models[method] = (cfg, model)
        del payload
    cfg = models['mean100_final800'][0]
    variance = TaylorH2VarianceModel(operator, **cfg['noise'])
    start = time.time()
    for c in cases:
        raw = input_item(c['path'], c['subset'], device)
        assert not {'measured_mean', 'measured_variance', 'ground_truth'} & raw.keys()
        predictions = {}
        for method in METHODS:
            folder = root / 'evaluation' / method / c['id']
            marker = folder / 'complete.json'
            if marker.exists():
                saved = json.loads(marker.read_text())
                assert saved['input_fingerprint'] == manifest['input_fingerprints'][c['id']]
                assert SHA(folder / 'reconstruction.npy') == saved['prediction_sha256']
                if method in models:
                    assert saved['checkpoint_sha256'] == manifest['checkpoints'][method]['sha256']
                continue
            extras, volumes = {}, {}
            with torch.inference_mode():
                if method not in models:
                    pred = raw['g_mean'] if method == 'mean_rl3' else raw['f_var']
                else:
                    current, model = models[method]
                    out, beta0 = exp.forward(model, raw, operator, None, current)
                    pred = out.reconstruction
                    anchor = out._physical_anchor
                    correction = out._effective_correction
                    extras = dict(beta0=float(beta0), beta=float(out.beta), gain=float(out._gain),
                                  correction_to_anchor_l2=float(correction.norm()/anchor.norm().clamp_min(1e-30)))
                    volumes = {'physical_anchor': anchor[0, 0].cpu().numpy(),
                               'effective_correction': correction[0, 0].cpu().numpy(),
                               'pre_gain_reconstruction': out._pre_gain_reconstruction[0, 0].cpu().numpy()}
                    if c['id'] == 'V01_subset_01':
                        contaminated = dict(raw, ground_truth=torch.full_like(pred, 999),
                                            measured_mean=torch.full_like(raw['input_mean'], 999),
                                            measured_variance=torch.full_like(raw['input_mean'], 999))
                        repeated, _ = exp.forward(model, contaminated, operator, beta0, current)
                        error = float((pred - repeated.reconstruction).norm()/pred.norm().clamp_min(1e-30))
                        assert error < 1e-6
                        extras['target_gt_contamination_relative_l2'] = error
                        reference = Path(manifest['checkpoints'][method]['path']).parent / f"validation/step_{manifest['checkpoints'][method]['step']:06d}/V01/subset_01/reconstruction.npy"
                        saved = torch.from_numpy(np.load(reference)).to(device)
                        error = float((pred[0, 0] - saved).norm()/saved.norm().clamp_min(1e-30))
                        assert error < 1e-4, (method, error)
                        extras['frozen_validation_regression_relative_l2'] = error
                predictions[method] = (pred.clone(), extras, volumes)
        # Targets are loaded after all network inference for this case.
        target = ev._targets(c, device)
        for method, (pred, extras, volumes) in predictions.items():
            array = pred[0, 0].cpu().numpy().astype(np.float32)
            assert array.shape == (10, 260, 260) and np.isfinite(array).all() and array.min() >= 0
            scores = ev._common_scores(pred, target['measured_mean'], target['measured_variance'], operator, variance, cfg)
            metadata = dict(method=method, case_id=c['id'], sample_id=c['sample'], subset=c['subset'],
                            split=c['split'], checkpoint_role='best' if method.endswith('740') else 'final',
                            weight_step=manifest['checkpoints'].get(method, {}).get('step', ''), **extras, **scores)
            if 'ground_truth' in target:
                metadata.update(ev._structure_row(array, target['ground_truth'][0, 0].cpu().numpy()))
            else:
                metadata.update(prediction_max=float(array.max()), prediction_sum=float(array.sum()))
                assert array.max() == 0, (method, 'zero input hallucination')
            folder = root / 'evaluation' / method / c['id']
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / 'reconstruction.npy', array)
            save_volume_tiff(folder / 'reconstruction.tif', array)
            for key, value in volumes.items():
                np.save(folder / f'{key}.npy', value.astype(np.float32))
            write_json(folder / 'complete.json', dict(complete=True, metrics=metadata,
                       input_fingerprint=manifest['input_fingerprints'][c['id']],
                       checkpoint_sha256=manifest['checkpoints'].get(method, {}).get('sha256'),
                       prediction_sha256=SHA(folder / 'reconstruction.npy'), inference_target_or_gt_used=False))
        print(json.dumps(dict(case=c['id'], finished=cases.index(c)+1, total=94,
                              seconds=round(time.time()-start), peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)), flush=True)
    write_json(root / 'inference_complete.json', dict(complete=True, unique_network_predictions=282,
               rl_controls_reused=188, total_cases=470, peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30))


def local_tables(root):
    manifest, cases = setup(root)
    assert (root / 'inference_complete.json').exists()
    tables = {k: [] for k in ('metrics', 't02_tubes', 't03_lines', 't04_axial', 'v03_beads',
                              'fixed_profiles', 'point_targets', 'point_pairs', 'priority_lines', 'depth_profiles', 'axial_points')}
    _, priority_cfg = load_config()
    functions = dict(T02=('t02_tubes', local.t02), T03=('t03_lines', local.t03),
                     T04=('t04_axial', local.t04), V03=('v03_beads', local.v03))
    for c in cases:
        target = ev._targets(c, torch.device('cpu'))
        truth = target['ground_truth'][0, 0].numpy() if 'ground_truth' in target else None
        predictions = {}
        for method in METHODS:
            folder = root / 'evaluation' / method / c['id']
            rec = json.loads((folder / 'complete.json').read_text())
            assert SHA(folder / 'reconstruction.npy') == rec['prediction_sha256']
            pred = np.load(folder / 'reconstruction.npy')
            predictions[method] = pred
            tables['metrics'].append(rec['metrics'])
            meta = dict(method=method, case_id=c['id'], sample_id=c['sample'], subset=c['subset'], split=c['split'])
            if c['sample'] in functions:
                name, fn = functions[c['sample']]
                rows, profiles = fn(pred, truth, meta)
                tables[name].extend(rows)
                tables['fixed_profiles'].extend(profiles)
        if c['split'] == 'priority':
            if c['family'] == 'points':
                pts, pairs, profiles = ev.pa._point_metrics(priority_cfg, c['scene_id'], c['repeat'], truth, predictions)
                tables['point_targets'].extend(pts); tables['point_pairs'].extend(pairs); tables['depth_profiles'].extend(profiles)
            elif c['family'] == 'lines':
                tables['priority_lines'].extend(ev.pa._line_metrics(priority_cfg, c['scene_id'], c['repeat'], predictions))
            else:
                pts, pairs = ev.pa._axial_metrics(priority_cfg, c['repeat'], predictions)
                tables['axial_points'].extend(pts); tables['point_pairs'].extend(pairs)
        print('LOCAL', c['id'], flush=True)
    checks = {}
    for sample in ('T03', 'T04', 'V03'):
        c = next(c for c in cases if c['sample'] == sample)
        truth = ev._targets(c, torch.device('cpu'))['ground_truth'][0, 0].numpy()
        rows, _ = functions[sample][1](truth, truth, {'method': 'GT_identity'})
        rows = [r for r in rows if r['threshold_fraction'] == .1]
        if sample == 'T03': assert all(r['separated'] for r in rows)
        elif sample == 'V03': assert all(r['detected'] for r in rows)
        else: assert all(r['separated'] for r in rows if r['separation_applicable'])
        checks[sample] = dict(GT_positive_control_passed=True, items=len(rows))
    for name, rows in tables.items():
        write_csv(root / 'analysis' / f'{name}.csv', rows)
    write_json(root / 'local_complete.json', dict(complete=True, tables={k: len(v) for k,v in tables.items()}, checks=checks))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['setup', 'evaluate', 'local'])
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    {'setup': setup, 'evaluate': evaluate, 'local': local_tables}[args.action](args.output)
