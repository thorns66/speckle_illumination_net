"""Frozen diagnostic only: trace empty/low-signal outputs without editing the model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--output-dir', default='outputs/background_diagnosis_20260907')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    inventory = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    gpu = next(line.split(',') for line in inventory.splitlines()
               if int(line.split(',')[0]) == args.gpu)
    if int(gpu[2]) > 1024 or int(gpu[3]) > 10:
        raise RuntimeError('Selected GPU is busy; no diagnostic launched')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu[1].strip()
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    import numpy as np
    import torch
    from datasets.matlab_multivolume_dataset import load_inference_input
    from train_volume import _model_from_config
    from models.blocks import normalize_feature_input

    destination = root / args.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'diagnosis.json').exists():
        raise FileExistsError('Completed diagnostic exists; choose a new output directory')
    source = root / 'outputs/priority_validation_20260907/generated'
    checkpoint_path = root / 'outputs/multivolume_n10_no_mean_run01/checkpoint_best.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert checkpoint['best_step'] == 160
    config = checkpoint['config']
    torch.set_num_threads(4)
    torch.manual_seed(20260907)
    device = torch.device('cuda:0')
    model = _model_from_config(config).to(device).eval()
    model.load_state_dict(checkpoint['model_state'])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tensor_names = ('f_var', 'f_var_feature', 'g_mean', 'residual_frames', 'z_values_um')

    def load(name):
        raw = load_inference_input(source / name, 1, var_feature_representation='sqrt')
        item = {key: torch.from_numpy(np.asarray(raw[key])).float().unsqueeze(0).to(device)
                for key in tensor_names}
        contract = json.loads((source / name / 'network/inference_contract.json').read_text())
        item['beta0'] = torch.tensor([contract['beta0']], device=device)
        return item

    def run(item):
        return model(item['f_var'], item['g_mean'], item['residual_frames'], item['z_values_um'],
                     var_feature_volume=item['f_var_feature'], beta0=item['beta0'])

    def stats(value):
        a = value.detach().float()
        return {'max': float(a.max()), 'min': float(a.min()), 'sum': float(a.sum()),
                'rms': float(a.square().mean().sqrt()),
                'positive_fraction': float((a > 0).float().mean())}

    def write_csv(name, rows):
        with (destination / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    rows = []
    features = {}
    zero = load('zero_control')
    assert all(torch.count_nonzero(zero[k]) == 0 for k in tensor_names if k != 'z_values_um')
    features['normalized_var_input'] = stats(normalize_feature_input(zero['f_var'], (-3,-2,-1)))
    features['normalized_mean_input'] = stats(normalize_feature_input(zero['g_mean'], (-3,-2,-1)))
    base_flags = {key: getattr(model, key) for key in (
        'use_variance_branch','use_mean_branch','use_set_branch','use_network_refinement')}
    cases = ['baseline', 'no_refinement', 'no_set', 'zero_z_coordinate',
             'all_input_branches_off', 'zero_fused_features', 'zero_final_head_bias',
             'zero_all_conv_bias', 'zero_all_groupnorm_bias', 'zero_all_affine_bias',
             'zero_all_affine_bias_and_z']
    with torch.inference_mode():
        base = run(zero)
        base_array = base.reconstruction[0,0].cpu().numpy()
        saved = np.load(source / 'zero_control/network/reconstruction.npy')
        reproduction = float(np.linalg.norm(base_array-saved) / max(np.linalg.norm(saved),1e-12))
        np.save(destination / 'zero_baseline.npy', base_array)
        features['zero_anchor'] = stats(base.beta[:,None,None,None,None] * zero['f_var'])
        features['zero_residual'] = stats(base.residual)
        features['positive_map_equals_relu_residual_max_error'] = float(
            (base.reconstruction-base.residual.clamp_min(0)).abs().max())
        for prefix, pyramid in [('variance',base.variance_features),('mean',base.mean_features),
                                ('set',base.set_features)]:
            for i, feature in enumerate(pyramid): features[f'{prefix}_{i}'] = stats(feature)
        features['final_head_bias'] = float(model.decoder.head.bias[0])
        features['head_weight_part'] = stats(base.residual-model.decoder.head.bias[0])
        del base
        for case in cases:
            model.load_state_dict(checkpoint['model_state'])
            for key, value in base_flags.items(): setattr(model, key, value)
            item = dict(zero)
            hooks = []
            if case == 'no_refinement': model.use_network_refinement = False
            if case == 'no_set': model.use_set_branch = False
            if case in ('zero_z_coordinate','zero_all_affine_bias_and_z'):
                item['z_values_um'] = torch.zeros_like(item['z_values_um'])
            if case == 'all_input_branches_off':
                model.use_variance_branch = model.use_mean_branch = model.use_set_branch = False
            if case == 'zero_fused_features':
                hooks.append(model.decoder.register_forward_pre_hook(
                    lambda module, inputs: (tuple(torch.zeros_like(x) for x in inputs[0]),)))
            if case == 'zero_final_head_bias': model.decoder.head.bias.zero_()
            for module in model.modules():
                is_conv = isinstance(module,(torch.nn.Conv2d,torch.nn.Conv3d))
                is_norm = isinstance(module,torch.nn.GroupNorm)
                if ((case == 'zero_all_conv_bias' and is_conv)
                    or (case == 'zero_all_groupnorm_bias' and is_norm)
                    or (case in ('zero_all_affine_bias','zero_all_affine_bias_and_z')
                        and (is_conv or is_norm))):
                    if module.bias is not None: module.bias.zero_()
            out = run(item)
            rows.append({'case':case, **stats(out.reconstruction),
                         'residual_min':float(out.residual.min()),
                         'residual_max':float(out.residual.max())})
            for hook in hooks: hook.remove()
            print(json.dumps(rows[-1]), flush=True)
            del out
        model.load_state_dict(checkpoint['model_state'])
        for key, value in base_flags.items(): setattr(model, key, value)
        write_csv('zero_ablation.csv', rows)

        # Controlled network input scaling: hold the cached analytic beta0 fixed.
        # This is not a rerun of RL or a real photon-noise simulation.
        scale_rows = []
        for sample in ('points_z060_r01','lines_z060_r01'):
            item = load(sample)
            original = run(item).reconstruction.clone()
            historical = torch.from_numpy(np.load(source/sample/'network/reconstruction.npy')).to(device)
            features[f'{sample}_reproduction_relative_l2'] = float(
                (original[0,0]-historical).norm()/historical.norm().clamp_min(1e-12))
            for gain in (0.,1e-4,1e-3,.01,.1,.5,1.,2.):
                scaled = {k: v*gain if k in ('f_var','f_var_feature','g_mean','residual_frames') else v
                          for k,v in item.items()}
                out = run(scaled)
                expected = gain*original
                scale_rows.append({'sample':sample,'gain':gain, **stats(out.reconstruction),
                    'sum_over_expected':float(out.reconstruction.sum()/expected.sum()) if gain>0 else None,
                    'relative_l2_to_gain_times_original':float((out.reconstruction-expected).norm()/expected.norm())
                        if gain>0 else None})
                print(json.dumps(scale_rows[-1]), flush=True)
                del out
            del original, item
        write_csv('input_scaling.csv', scale_rows)

    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    report = {'complete': True, 'best_step':160,'checkpoint_sha256':digest,
              'zero_reproduction_relative_l2':reproduction,'features':features,
              'ablation':rows,'input_scaling':scale_rows,
              'scope':'frozen diagnosis; model source and checkpoint not modified',
              'scaling_scope':'scaled network inputs, fixed cached beta0; no new RL/noise simulation',
              'input_gt_and_holdout_read':False, 'config':config, 'gpu_inventory':inventory}
    (destination/'diagnosis.json').write_text(json.dumps(report,indent=2))
    print('COMPLETE',destination,flush=True)


if __name__ == '__main__':
    main()
