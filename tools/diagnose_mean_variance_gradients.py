"""Inspect loss gradients with respect to frozen volumes; performs no training."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--data-root', default='data/matlab_cells_pilot_v2_r04')
    parser.add_argument('--output-dir', default='outputs/background_diagnosis_20260907')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root/'tools')]
    inventory = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True)
    gpu = next(line.split(',') for line in inventory.splitlines()
               if int(line.split(',')[0])==args.gpu)
    if int(gpu[2])>1024 or int(gpu[3])>10: raise RuntimeError('GPU is busy')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu[1].strip()
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    destination = root/args.output_dir
    destination.mkdir(parents=True,exist_ok=True)
    if (destination/'loss_gradient_diagnosis.json').exists():
        raise FileExistsError('Use a new output path; diagnostic already completed')
    os.environ['MPLCONFIGDIR'] = str(destination/'mpl_cache')
    import numpy as np
    import torch
    from losses.self_supervised_losses import compute_self_supervised_loss
    from priority_validation_analysis import (
        _load_operator, _load_calibration, _read_yxz, _target_statistics, _torch_volume)

    torch.set_num_threads(4)
    device=torch.device('cuda:0')
    checkpoint=torch.load(root/'outputs/multivolume_n10_no_mean_run01/checkpoint_best.pt',
                          map_location='cpu',weights_only=False)
    config=checkpoint['config']
    source=root/'outputs/priority_validation_20260907'
    operator,variance_model=_load_operator(config,device)
    _,illum_var=_load_calibration(source)
    rows=[]
    for sample in ('P08','P09','P10'):
        truth=_read_yxz(root/args.data_root/sample/'prepared.mat','ground_truth')
        volumes={'calibrated_truth':truth*np.sqrt(np.maximum(illum_var,0)),
                 'network_subset01':np.load(source/'existing_inference'/f'{sample}_subset_01'/'reconstruction.npy')}
        mu,var=_target_statistics(source,sample,1)
        mean=torch.from_numpy(mu).to(device)[None,None]
        variance=torch.from_numpy(var).to(device)[None,None]
        for name,array in volumes.items():
            g=_torch_volume(array,device).requires_grad_(True)
            terms=compute_self_supervised_loss(g,mean,variance,operator,variance_model,
                      **config['loss'],physics_use_checkpoint=True)
            gm=torch.autograd.grad(terms.normalized_mean,g,retain_graph=True)[0]
            gv=torch.autograd.grad(terms.normalized_var,g,retain_graph=True)[0]
            gt=torch.autograd.grad(terms.weighted_tv,g)[0]
            mn,vn,tn=gm.norm(),gv.norm(),gt.norm()
            # Separate global brightness direction from shape directions in volume space.
            axis=g.detach()/g.detach().norm().clamp_min(1e-30)
            dm=(gm*axis).sum(); dv=(gv*axis).sum()
            sm=gm-dm*axis; sv=gv-dv*axis
            row={'sample_id':sample,'candidate':name,'target_repeat':1,
                 'mean_loss':float(terms.normalized_mean.detach()),
                 'variance_loss':float(terms.normalized_var.detach()),
                 'grad_mean_norm':float(mn),'grad_variance_norm':float(vn),
                 'grad_mean_to_variance_norm':float(mn/vn.clamp_min(1e-30)),
                 'gradient_cosine':float((gm*gv).sum()/(mn*vn).clamp_min(1e-30)),
                 'shape_gradient_cosine':float((sm*sv).sum()/(sm.norm()*sv.norm()).clamp_min(1e-30)),
                 'd_mean_d_log_gain':float((gm*g.detach()).sum()),
                 'd_variance_d_log_gain':float((gv*g.detach()).sum()),
                 'weighted_tv_gradient_to_variance':float(tn/vn.clamp_min(1e-30))}
            rows.append(row)
            print(json.dumps(row),flush=True)
            del terms,gm,gv,gt,g,sm,sv
    with (destination/'loss_gradients.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    result={'complete':True,'rows':rows,
            'scope':'output-volume gradients, not shared-network parameter gradients; no optimizer step',
            'sampling':'P08/P09/P10; first new independent 90-frame target; existing subset 01',
            'config':config['loss']}
    (destination/'loss_gradient_diagnosis.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__': main()
