"""Diagnostic variant: fit one output gain from input ten frames, never from GT."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--data-root',default='data/matlab_cells_pilot_v2_r04')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(root),str(root/'tools')]
    output=root/'outputs/background_diagnosis_20260907/mean_gain_variant'
    if (output/'complete.json').exists(): raise FileExistsError('Diagnostic already complete')
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
                                      '--format=csv,noheader,nounits'],text=True)
    gpu=next(x.split(',') for x in inventory.splitlines() if int(x.split(',')[0])==args.gpu)
    if int(gpu[2])>1024 or int(gpu[3])>10: raise RuntimeError('GPU busy')
    os.environ['CUDA_VISIBLE_DEVICES']=gpu[1].strip()
    os.environ['MPLCONFIGDIR']=str(output/'mpl_cache')
    output.mkdir(parents=True,exist_ok=True)
    import numpy as np
    import torch
    from priority_validation_analysis import _load_operator,_score_volume,_target_statistics,_read_yx
    torch.set_num_threads(4)
    config=torch.load(root/'outputs/multivolume_n10_no_mean_run01/checkpoint_best.pt',
                      map_location='cpu',weights_only=False)['config']
    device=torch.device('cuda:0')
    operator,variance_model=_load_operator(config,device)
    original=root/'outputs/priority_validation_20260907'
    rows=[]
    for sample in ('P08','P09','P10'):
        g=np.load(original/'existing_inference'/f'{sample}_subset_01'/'reconstruction.npy')
        mu90,var90=_target_statistics(original,sample,1)
        mu10=_read_yx(root/args.data_root/sample/'subsets/subset_01.mat',
                     'input_physics_mean_float')
        base,muhat,vhat=_score_volume(g,mu90,var90,operator,variance_model,config['loss'],device)
        p=np.asarray(muhat,dtype=np.float64).ravel();t=np.asarray(mu10,dtype=np.float64).ravel()
        gain=max(float(p@t)/max(float(p@p),1e-30),0.)
        projected=(gain*g).astype(np.float32)
        np.save(output/f'{sample}_subset01.npy',projected)
        scale=max(float(np.abs(mu90).mean()),1e-8)
        lm=float(torch.nn.functional.smooth_l1_loss(torch.from_numpy(gain*muhat/scale),
                                                   torch.from_numpy(mu90/scale)))
        lv=float(torch.nn.functional.smooth_l1_loss(
            torch.log(torch.from_numpy(gain**2*vhat)+config['loss']['var_log_eps']),
            torch.log(torch.from_numpy(var90)+config['loss']['var_log_eps'])))
        shape_error=float(np.max(np.abs(g/g.sum()-projected/projected.sum())))
        row={'sample_id':sample,'subset':1,'target_repeat':1,'gain_fitted_from_input10':gain,
             'mean_loss_before':base['normalized_mean_loss'],'mean_loss_after':lm,
             'variance_loss_before':base['normalized_var_loss'],'variance_loss_after':lv,
             'normalized_volume_max_difference':shape_error}
        rows.append(row);print(json.dumps(row),flush=True)
    z=np.load(original/'generated/zero_control/network/reconstruction.npy')
    # The zero input mean has zero dot product with every finite prediction.
    zero_projection=z*0.
    np.save(output/'zero_control.npy',zero_projection)
    with (output/'scores.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (output/'complete.json').write_text(json.dumps({'complete':True,'rows':rows,
        'zero_max':float(zero_projection.max()),'zero_sum':float(zero_projection.sum()),
        'gain_source':'input10 measured mean only; one global nonnegative LS gain',
        'scope':'new diagnostic variant; original official predictions not changed',
        'limitations':'fixes global scale only, not false local structure or approximate variance physics'},indent=2))


if __name__=='__main__':main()
