"""Supplementary acceptance against actual trained V3 checkpoint states."""
from __future__ import annotations
import copy,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import h5py,numpy as np,torch,yaml
from tools import v3_compare_experiment as exp
from tools import three_way_experiment as old
from tools.three_way_checks import input_item
from tools.priority_validation_analysis import _read_yxz,_read_yx
from losses.self_supervised_losses import TaylorH2VarianceModel

def relative(a,b):
    return float((a-b).norm()/b.norm().clamp_min(1e-30))

def run():
    torch.set_num_threads(4);torch.cuda.set_device(0);exp.configure_precision()
    device=torch.device('cuda:0')
    config=yaml.safe_load((exp.OUTPUT/'e3.yaml').read_text())
    operator=exp.load_operator(config,device)
    variance=TaylorH2VarianceModel(operator)
    data=input_item(exp.DATA/'V01',1,device)
    from datasets.matlab_multivolume_dataset import DatasetItemKey,_read_targets
    target=_read_targets(DatasetItemKey('V01',1,'validation',exp.DATA/'V01'),include_ground_truth=False)
    data.update({k:torch.from_numpy(v).float()[None].to(device) for k,v in target.items() if k in ('measured_mean','measured_variance')})
    results={}
    # Check a real old MATLAB sensor frame against the frozen Python operator.
    scene=old.PRIORITY/'generated/points_z020_r01'
    truth=_read_yxz(scene/'prepared.mat','ground_truth')
    with h5py.File(old.PRIORITY/'shared_illumination/repeat_01.mat') as handle:
        illum=np.asarray(handle['illumination_raw'][0]).transpose(0,2,1).copy()
    matlab=_read_yx(scene/'sensor_frames/frame_001.mat','sensor_pre_detector')
    with torch.inference_mode():
        py=operator(torch.from_numpy(truth*illum)[None,None].to(device))[0,0]
    error=relative(py,torch.from_numpy(matlab).to(device))
    assert error<=1e-4,error
    results['python_matlab_forward_relative_l2']=error
    for arm in ('baseline','e3','e3_mean005'):
        payload=torch.load(exp.OUTPUT/arm/'checkpoint_step_000200.pt',map_location='cpu',weights_only=False)
        cfg=copy.deepcopy(payload['config'])
        cfg['experiment']['output_dir']='/tmp/v3_supplemental_gradient_checks/'+arm
        model=exp.build_model(cfg).to(device).train()
        model.load_state_dict(payload['model_state'])
        exp._STATE.update(completed_steps=200,phase='supplemental_check',evaluation=False)
        optimizer=exp._optimizer(model,cfg)
        optimizer.load_state_dict(payload['optimizer_state'])
        terms=None
        if arm!='baseline':
            out,_=exp.forward(model,data,operator,config=cfg)
            terms=exp.loss(out,data,operator,variance,cfg)
            mean_to_q=torch.autograd.grad(terms.weighted_mean,out._shape,allow_unused=True,retain_graph=True)[0]
            variance_to_gamma=torch.autograd.grad(terms.weighted_var,model.mean_gain_gamma,allow_unused=True,retain_graph=True)[0]
            assert mean_to_q is None and variance_to_gamma is None
            results[arm+'_ordinary_mean_to_q_is_none']=True
            results[arm+'_variance_to_gamma_is_none']=True
            del out,terms
        # Same actual checkpoint + restored Adam + identical next update twice.
        # No experiment files or training counters are modified by this audit.
        states=[]
        for repeat in range(2):
            model.load_state_dict(payload['model_state'])
            optimizer.load_state_dict(copy.deepcopy(payload['optimizer_state']))
            exp._STATE.update(completed_steps=200,phase='supplemental_check',evaluation=False)
            torch.manual_seed(987);torch.cuda.manual_seed_all(987)
            optimizer.zero_grad(set_to_none=True)
            out,_=exp.forward(model,data,operator,config=cfg)
            terms=exp.loss(out,data,operator,variance,cfg)
            terms.total.backward();optimizer.step()
            states.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        maxerr=max(relative(states[0][k],states[1][k]) for k in states[0] if states[0][k].is_floating_point())
        assert maxerr<=1e-4,(arm,maxerr)
        assert exp._STATE['completed_steps']==201
        assert optimizer.param_groups[0]['lr']==1e-4
        results[arm+'_restored_adam_next_update_max_relative_error']=maxerr
        if arm=='e3_mean005':
            ramps={}
            model.load_state_dict(payload['model_state'])
            for step in (49,50):
                exp._STATE.update(completed_steps=step-1,phase='supplemental_check',evaluation=False)
                out,_=exp.forward(model,data,operator,config=cfg)
                terms=exp.loss(out,data,operator,variance,cfg)
                m=terms.scalar_metrics()
                budget=.05*min(step/50,1)
                assert m['mean_shape_gradient_ratio']<=budget+1e-7
                ramps[step]={'budget':budget,'actual_ratio':m['mean_shape_gradient_ratio']}
            results['actual_step49_step50_gradient_check']=ramps
        del model,optimizer,payload,out,terms
        torch.cuda.empty_cache()
    results['passed']=True
    old.write_json(exp.OUTPUT/'supplemental_gpu_acceptance.json',results)
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':run()
