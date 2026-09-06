"""Frozen-state one-step optimizer audit, NOT a new reconstruction run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from physics.finite_phase_cs import DenseFinitePhaseCs
from tools.diagnose_covariance_oracle_provenance import compute_actions
from tools.diagnose_cs_information import load_operator, model_action, vector_loss
from tools.diagnose_fixed_depth_lateral import bounded_anchor_shape
from tools.scan_covariance_sketch_depth import _empirical_action, _lowpass_probes


def shape_tangent(shape, raw, direction, bound):
    log_tangent = (1-torch.tanh(raw/bound).square())*direction
    return shape*(log_tangent-(shape*log_tangent).sum())


def optimizer_direction(gradient, state, group, variant):
    """Side-effect-free Adam and counterfactual directions with shared history."""
    if group.get('weight_decay', 0) or group.get('amsgrad', False) or group.get('maximize', False):
        raise ValueError('Audit supports the actual unregularized standard Adam only')
    beta1, beta2 = group['betas']
    step = int(state['step'])+1
    old_m = state['exp_avg'].to(gradient)
    old_v = state['exp_avg_sq'].to(gradient)
    m = torch.lerp(old_m, gradient, 1-beta1)/(1-beta1**step)
    v = (old_v*beta2 + gradient.square()*(1-beta2))/(1-beta2**step)
    scale = v.sqrt()
    if variant == 'adam':
        denominator = scale+group['eps']
        direction = -group['lr']*m/denominator
    elif variant == 'damped_adam':
        denominator = scale+max(float(scale.median()), group['eps'])
        direction = -group['lr']*m/denominator
    elif variant == 'isotropic_momentum':
        denominator = scale.square().mean().sqrt()+group['eps']
        direction = -group['lr']*m/denominator
    elif variant == 'gd':
        denominator = torch.ones((), device=gradient.device, dtype=gradient.dtype)
        direction = -gradient
    else:
        raise ValueError(variant)
    if not torch.isfinite(direction).all():
        raise FloatingPointError('Nonfinite optimizer direction')
    return direction, denominator


def cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float((a@b)/(a.norm()*b.norm()).clamp_min(1e-30))


def report_image(shape, truth):
    lap = (shape[1:-1, 2:]+shape[1:-1, :-2]+shape[2:, 1:-1]
           +shape[:-2, 1:-1]-4*shape[1:-1, 1:-1])
    return {'relative_l2': float((shape-truth).norm()/truth.norm()),
            'peak_ratio': float(shape.max()/truth.max()),
            'laplacian_norm': float(lap.norm()/shape.norm())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--root', default='outputs/linear_float_oracle_50um')
    args = parser.parse_args()
    output, root = Path(args.output).resolve(), Path(args.root).resolve()
    if output.exists(): raise FileExistsError(output)
    saved = torch.load(root/'exact_population_curvature/r003/state_step019.pt',
                       map_location='cpu', weights_only=True)
    if saved['step'] != 19 or saved['curvature_calibration'] is not None:
        raise ValueError('Must be the common, unregularized 20-update warmup state')
    settings = saved['arguments']
    if settings['train_probe_batch'] != 32 or settings['bound'] != 2 or settings['cs_type'] != 'finite_phase':
        raise ValueError('Wrong shared warmup experiment')
    dataset = root/'dataset'
    metadata = json.loads((dataset/'metadata.json').read_text())
    if metadata['per_frame_sensor_normalization'] or metadata['quantization']:
        raise ValueError('Linear float simulation required')
    device = torch.device(args.device); torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    operator = load_operator(Path(settings['config']).resolve(), device)
    cs = DenseFinitePhaseCs.load(root/'exact_population_cs', device=device,
                                system_mean=metadata['speckle_system_mean_before_scaling'])
    anchor = torch.from_numpy(np.load(dataset/'anchor_rl3.npy')[4]).to(device)
    anchor = anchor.clamp_min(anchor.max()*1e-8); anchor /= anchor.sum()
    truth = torch.from_numpy(np.load(dataset/'target.npy')[4]).to(device); truth /= truth.sum()
    group = saved['optimizer']['param_groups'][0]
    saved_state = saved['optimizer']['state'][group['params'][0]]
    banks = {}
    for name, count, seed in (('train', 32, 20261010), ('holdout', 128, 20261030)):
        q_np = _lowpass_probes(count, (260,260), sigma=0, seed=seed)
        q = torch.from_numpy(q_np).to(device)
        with torch.no_grad():
            back = torch.cat([operator.adjoint(q[start:start+4,None]) for start in range(0,count,4)])
            population = compute_actions(truth, back, cs, operator)
        frames = np.load(dataset/f'{name}_frames.npy', mmap_mode='r')
        empirical = torch.from_numpy(_empirical_action(frames, q_np)).to(device)
        banks[name] = (back, population, empirical)
    raw20 = saved['raw'].to(device)
    reference20 = torch.from_numpy(np.load(root/'exact_population_fullbatch/b2/reconstruction_step019.npy')[4]).to(device)
    torch.testing.assert_close(bounded_anchor_shape(anchor,raw20,2)[0], reference20, rtol=1e-5, atol=1e-10)
    empty_state = {'step': 0, 'exp_avg': torch.zeros_like(anchor), 'exp_avg_sq': torch.zeros_like(anchor)}
    states = [('initial', torch.zeros_like(anchor), empty_state, ('population','empirical')),
              ('shared_step19', raw20, saved_state, ('population',))]
    rows = []; baselines = {}
    for name, current_raw, optimizer_state, target_types in states:
        raw = current_raw.detach().clone().requires_grad_()
        initial_shape = bounded_anchor_shape(anchor,raw,2)[0].detach()
        with torch.no_grad():
            initial_hold = compute_actions(initial_shape,banks['holdout'][0],cs,operator)
            baselines[name] = {**report_image(initial_shape,truth),
                'population_holdout': float(vector_loss(initial_hold,banks['holdout'][1])[0]),
                'empirical_holdout': float(vector_loss(initial_hold,banks['holdout'][2])[0])}
        for target_type in target_types:
            gradient = torch.zeros_like(raw); loss_value = 0.
            back, population, empirical = banks['train']
            target = population if target_type == 'population' else empirical
            for start in range(0,len(back),4):
                shape = bounded_anchor_shape(anchor,raw,2)[0]
                prediction = model_action(shape,back[start:start+4],cs,operator)
                loss = vector_loss(prediction,target[start:start+4])[0]
                weight = len(prediction)/len(back)
                gradient += torch.autograd.grad(loss,raw)[0].detach()*weight
                loss_value += float(loss.detach())*weight
            adam_delta = optimizer_direction(gradient,optimizer_state,group,'adam')[0]
            reference_norm = shape_tangent(initial_shape,raw.detach(),adam_delta,2).norm()
            if reference_norm <= 0: raise FloatingPointError('Zero reference step')
            for variant in ('adam','isotropic_momentum','damped_adam','gd'):
                delta, denominator = optimizer_direction(gradient,optimizer_state,group,variant)
                tangent = shape_tangent(initial_shape,raw.detach(),delta,2)
                multiplier = reference_norm/tangent.norm().clamp_min(1e-30)
                delta = delta*multiplier
                tangent = shape_tangent(initial_shape,raw.detach(),delta,2)
                with torch.no_grad():
                    candidate = bounded_anchor_shape(anchor,raw.detach()+delta,2)[0]
                    hold = compute_actions(candidate,banks['holdout'][0],cs,operator)
                    row = {'state': name, 'gradient_target': target_type, 'variant': variant,
                           'train_loss_before': loss_value,
                           'population_holdout': float(vector_loss(hold,banks['holdout'][1])[0]),
                           'empirical_holdout': float(vector_loss(hold,banks['holdout'][2])[0]),
                           'image_tangent_norm': float(tangent.norm()),
                           'actual_image_change_norm': float((candidate-initial_shape).norm()),
                           'raw_step_norm': float(delta.norm()), 'step_multiplier': float(multiplier),
                           'truth_error_alignment': cos(tangent,truth-initial_shape),
                           'data_descent_inner_product': float((gradient*delta).sum()),
                           'inverse_denominator_quantiles': torch.quantile(1/denominator.flatten(),
                               torch.tensor([0.,.5,.9,.99,1.],device=device)).cpu().tolist(),
                           **report_image(candidate,truth)}
                rows.append(row); print(json.dumps(row),flush=True)
    report = {'complete': True, 'arguments': vars(args), 'baselines': baselines, 'rows': rows,
              'scope': 'Frozen-state counterfactual one-step audit; no iterative reconstruction or saved-image replacement.',
              'fairness': 'All directions matched to default Adam in image tangent L2 norm; no truth-dependent step choice.',
              'limits': 'Population directions use oracle targets. Empirical directions only at initial state with zero moments. Warmup moments came from population training. Two states cannot prove trajectory or real-data improvement.',
              'production_changed': False, 'mean_backprop_enabled': False}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
