"""Score completed 200-step population-Cs curvature diagnostics; never select by GT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tools.summarize_nested_frame_count_cs import evaluate


def verified_summary(directory: Path, *, curvature: float | None = None) -> dict:
    summary = json.loads((directory/'summary.json').read_text())
    steps = [row['step'] for row in summary['history']]
    expected = [0, *range(19, 200, 20)]
    if summary['steps'] != 200 or steps != expected:
        raise ValueError(f'Not a complete 200-update run: {directory}')
    args = summary['arguments']
    required = {'cs_type': 'finite_phase', 'target_link': 'matched', 'bound': 2.0,
                'probe_sigma': 0.0, 'train_probe_batch': 32, 'seed': 20260901}
    if any(args[key] != value for key, value in required.items()):
        raise ValueError(f'Incompatible physical/optimization control: {directory}')
    values = [(-1, summary['initial_holdout_loss'])] + [
        (row['step'], row['holdout_loss']) for row in summary['history']]
    selected = min(values, key=lambda item: item[1])
    if (summary['best_step'], summary['best_holdout_loss']) != selected:
        raise ValueError(f'Checkpoint was not selected by holdout loss: {directory}')
    if curvature is not None:
        calibration = summary['curvature_calibration']
        if args['curvature_gradient_ratio'] != curvature or calibration['step'] != 20:
            raise ValueError('Unexpected curvature calibration')
        expected_weight = curvature*calibration['data_gradient_norm']/calibration['curvature_gradient_norm']
        if not np.isclose(calibration['weight'], expected_weight, rtol=1e-10, atol=0):
            raise ValueError('Curvature weight formula mismatch')
        for row in summary['history']:
            expected_weight_at_step = 0 if row['step'] < 20 else expected_weight
            if row['curvature_weight'] != expected_weight_at_step:
                raise ValueError('Weight changed after initialization or started too early')
    return summary


def load_volume(path: Path, reference: np.ndarray) -> np.ndarray:
    volume = np.load(path)
    if volume.shape != reference.shape or not np.isfinite(volume).all() or np.any(volume < 0):
        raise ValueError(f'Invalid volume: {path}')
    if np.count_nonzero(np.delete(volume, 4, axis=0)) or not np.isclose(volume.sum(), 1, rtol=2e-6):
        raise ValueError(f'Not fixed-layer unit mass: {path}')
    return volume


def loss_matched_controls(root: Path, truth: np.ndarray) -> list[dict]:
    """Diagnostics only: match physical holdout residual, not image quality."""
    matches = []
    for bound, tag in ((0.5, "b0p5"), (2.0, "b2")):
        single_dir = root/f"exact_population_diagnostics/oracle_{tag}"
        full_dir = root/f"exact_population_fullbatch/{tag}"
        single = json.loads((single_dir/"summary.json").read_text())
        full = json.loads((full_dir/"summary.json").read_text())
        if single["steps"] != 200 or full["steps"] != 200:
            raise ValueError("Loss matching requires completed controls")
        target = single["best_holdout_loss"]
        closest = min(full["history"], key=lambda row: abs(np.log(row["holdout_loss"]/target)))
        match = {"bound": bound, "reference_step": single["best_step"],
                 "reference_holdout_loss": target, "matched_step": closest["step"],
                 "matched_holdout_loss": closest["holdout_loss"],
                 "relative_loss_mismatch": closest["holdout_loss"]/target-1,
                 "selection": "Closest log holdout covariance loss, no image metrics used"}
        for key, path in (
            ("single_probe_best", single_dir/"reconstruction_best.npy"),
            ("full32_loss_matched", full_dir/f"reconstruction_step{closest['step']:03d}.npy"),
            ("full32_final", full_dir/"reconstruction_final.npy"),
        ):
            match[key] = evaluate(load_volume(path, truth), truth)
        matches.append(match)
    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='outputs/linear_float_oracle_50um')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = root/'exact_population_curvature'
    truth = np.load(root/'dataset/target.npy')
    anchor = np.load(root/'dataset/anchor_rl3.npy')
    baseline = root/'exact_population_fullbatch/b2'
    control = verified_summary(baseline)
    directories = {'full32_r0': (baseline, None), 'full32_r003': (output/'r003', .003),
                   'full32_r01': (output/'r01', .01), 'full32_r03': (output/'r03', .03)}
    rows = [{'variant': 'anchor', 'checkpoint': 'initial', 'step': -1, **evaluate(anchor, truth)}]
    pictures = {'Truth': truth[4], 'RL3 anchor': anchor[4]}
    checks = {}
    for name, (directory, ratio) in directories.items():
        summary = verified_summary(directory, curvature=ratio)
        for key in ('dataset', 'config', 'cs_directory'):
            if summary['arguments'][key] != control['arguments'][key]:
                raise ValueError(f'Different source: {key}')
        if ratio is not None:
            warmup_errors = []
            for step in (0, 19):
                current = load_volume(directory/f'reconstruction_step{step:03d}.npy', truth)
                reference = load_volume(baseline/f'reconstruction_step{step:03d}.npy', truth)
                relative = float(np.linalg.norm(current-reference)/np.linalg.norm(reference))
                if relative > 1e-5:
                    raise ValueError('Warmup differs from shared unregularized control')
                warmup_errors.append(relative)
            checks[name] = {'warmup_relative_errors': warmup_errors,
                            'calibration': summary['curvature_calibration']}
        for checkpoint in ('best', 'final'):
            volume = load_volume(directory/f'reconstruction_{checkpoint}.npy', truth)
            row = {'variant': name, 'checkpoint': checkpoint,
                   'step': summary['best_step'] if checkpoint == 'best' else 199,
                   'holdout_loss': summary['best_holdout_loss'] if checkpoint == 'best' else summary['history'][-1]['holdout_loss'],
                   'elapsed_s': summary['history'][-1]['elapsed_s'],
                   'peak_gpu_memory_mb': summary['history'][-1]['peak_gpu_memory_mb'],
                   **evaluate(volume, truth)}
            rows.append(row)
            if checkpoint == 'best':
                pictures[name] = volume[4]
    single_path = root/'exact_population_diagnostics/oracle_b2/reconstruction_best.npy'
    single = load_volume(single_path, truth)
    pictures['single_probe_B2'] = single[4]
    rows.append({'variant': 'single_probe_B2', 'checkpoint': 'best', **evaluate(single, truth)})
    control_row = next(row for row in rows if row['variant'] == 'full32_r0' and row['checkpoint'] == 'best')
    comparisons = {}
    for row in rows:
        if row['checkpoint'] != 'best' or row['variant'] not in checks:
            continue
        comparisons[row['variant']] = {
            'ftc_improvement_fraction': 1-row['ftc_um']/control_row['ftc_um'],
            'l2_improvement_fraction': 1-row['unit_mass_relative_l2']/control_row['unit_mass_relative_l2'],
            'peak_reduction_fraction': 1-row['peak_ratio']/control_row['peak_ratio'],
            'annular_gradient_cosine_change': row['annular_gradient_cosine']-control_row['annular_gradient_cosine'],
            'radial_barb_reduction_fraction': 1-row['radial_barb_energy']/control_row['radial_barb_energy'],
        }
    mean_audit = json.loads((root/'exact_population_fullbatch/mean_forward_audit.json').read_text())
    mean_rows = [row for row in mean_audit['rows'] if row['mean_model']=='finite_source_exact'
                 and row['candidate'] in ('exact_b2_single', 'exact_b2_full32')]
    report = {'complete': True, 'rows': rows, 'checks': checks, 'comparisons_vs_full32_r0': comparisons,
              'preceding_forward_only_mean_check': mean_rows,
              'loss_matched_controls': loss_matched_controls(root, truth),
              'scope': 'Known 50 um, population targets from truth. Not real-data or unknown-depth validation.',
              'selection': 'Best by independent 16-probe covariance loss; report final too. GT metrics never select checkpoints.',
              'regularization': 'Hessian Frobenius mixed norm on raw pre-tanh log correction, warmup 20 steps, fixed weight.',
              'production_changed': False, 'mean_backprop_enabled': False,
              'caveat': 'These oracle data have no finite-frame covariance error. Good oracle results alone cannot establish empirical feasibility.'}
    (output/'comparison.json').write_text(json.dumps(report, indent=2))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for ax, (name, value) in zip(axes.flat, pictures.items()):
        value = value.astype(np.float64); value /= value.sum()
        ax.imshow(value[:140, :140], cmap='gray', vmin=0, vmax=1.5*truth[4].max()/truth[4].sum())
        label = 'anchor' if name == 'RL3 anchor' else name
        metric = next((row for row in rows if row['variant']==label and row['checkpoint'] in ('best','initial')), None)
        ax.set_title(name if metric is None else f"{name}\nFTC {metric['ftc_um']:.3f} um; L2 {metric['unit_mass_relative_l2']:.3f}")
        ax.axis('off')
    for ax in list(axes.flat)[len(pictures):]:
        ax.axis('off')
    fig.savefig(output/'upper_left_comparison.png', dpi=140)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
