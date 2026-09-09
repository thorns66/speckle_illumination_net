"""Score fixed GT/thickened/shifted candidates; no candidate gain fitting."""
import numpy as np
import torch
import torch.nn.functional as F
from tools.three_way_experiment import OUTPUT,DATA,PRIORITY,write_json
from tools.priority_validation_analysis import _read_yxz,_target_statistics,candidate_variants


def run(operator,correlated,old_variance):
    from tools.three_way_report import csv_write
    destination=OUTPUT/'analysis';destination.mkdir(parents=True,exist_ok=True)
    if (destination/'loss_audit_complete.json').exists():return
    moments=np.load(OUTPUT/'illumination/calibration_moments.npz')
    sigma=float(moments['sigma']);mean_field=torch.from_numpy(moments['mean'].astype(np.float32)).to(operator.H.device)[None,None]
    rows=[];gains=[]
    def score(pred,target,log=False):
        if log:return float(F.smooth_l1_loss(torch.log(pred.clamp_min(0)+1e-6),torch.log(target+1e-6)))
        scale=target.abs().mean().clamp_min(1e-8)
        return float(F.smooth_l1_loss(pred/scale,target/scale))
    with torch.inference_mode():
        for sample in ['P08','P09','P10']:
            truth=_read_yxz(DATA/sample/'prepared.mat','ground_truth')
            targets=[]
            for repeat in [1,2,3]:
                mean,var=_target_statistics(PRIORITY,sample,repeat)
                targets.append((repeat,torch.from_numpy(mean).to(operator.H.device)[None,None],
                                 torch.from_numpy(var).to(operator.H.device)[None,None]))
            for name,value in candidate_variants(truth).items():
                density=torch.from_numpy(np.ascontiguousarray(value)).float().to(operator.H.device)[None,None]
                g=density*sigma
                expected_mean=operator(density*mean_field)
                old_mean=operator(g)
                v_corr=correlated(g,expected_mean);v_old=old_variance(g,expected_mean)
                for repeat,mean,var in targets:
                    rows.append({'sample_id':sample,'candidate':name,'repeat':repeat,
                        'old_variance_loss':score(v_old,var,True),'corrected_variance_loss':score(v_corr,var,True),
                        'old_mean_loss':score(old_mean,mean),'calibrated_mean_loss':score(expected_mean,mean),
                        'fixed_density_to_output_scale':sigma,'gain_refitted':False})
                    if name=='truth':
                        for gain in [.25,.5,1.,2.,4.,8.]:
                            gains.append({'sample_id':sample,'repeat':repeat,'gain':gain,
                                'old_variance_loss':score(v_old*gain**2,var,True),
                                'corrected_variance_loss':score(v_corr*gain**2,var,True),
                                'old_mean_loss':score(old_mean*gain,mean),
                                'calibrated_mean_loss':score(expected_mean*gain,mean)})
    csv_write(destination/'fixed_candidate_loss_audit.csv',rows)
    csv_write(destination/'fixed_gt_gain_audit.csv',gains)
    write_json(destination/'loss_audit_complete.json',{'complete':True,'candidate_rows':len(rows),
         'gain_rows':len(gains),'source':'three new independent 90-frame targets per object',
         'used_for_checkpoint_selection':False,'density_scale':sigma,
         'corrected_mean_uses_independently_calibrated_illumination_mean':True})
