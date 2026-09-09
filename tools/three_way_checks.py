"""Numerical acceptance checks and independent illumination calibration."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from tools.three_way_experiment import (ROOT,OUTPUT,BASELINE,DATA,PRIORITY,build_model,
    load_operator,experiment_forward,experiment_loss,write_json,sha256,install_data)
from tools.three_way_physics import generate_bank,CorrelatedVariance
from datasets.matlab_multivolume_dataset import load_inference_input,MatlabMultiVolumeDataset
from training.multivolume_trainer import _to_device
from losses.self_supervised_losses import TaylorH2VarianceModel


def relative(a,b):
    return float((a-b).norm()/b.norm().clamp_min(1e-30))


def input_item(directory,subset=1,device='cuda:0'):
    raw=load_inference_input(directory,subset)
    names={'f_var','f_var_feature','g_mean','input_mean','residual_frames','z_values_um'}
    return {k:torch.from_numpy(v).float().unsqueeze(0).to(device) if k in names else v for k,v in raw.items()}


def gpu_checks():
    torch.set_num_threads(4);torch.cuda.set_device(0);install_data()
    config=yaml.safe_load((OUTPUT/'e1.yaml').read_text());device=torch.device('cuda:0')
    operator=load_operator(config,device)
    original=load_operator(config,device,sparse=False)
    item=input_item(DATA/'P09')
    start=time.monotonic()
    with torch.no_grad():
        g=item['f_var'];dense=original(g);sparse=operator(g)
        dense2=original.forward_squared(g.square());sparse2=operator.forward_squared(g.square())
        torch.manual_seed(1127);sensor=torch.rand_like(dense)
        lhs=(sparse*sensor).sum();rhs=(g*operator.adjoint(sensor)).sum()
        forward_error=relative(sparse,dense);square_error=relative(sparse2,dense2)
        adjoint_error=float((lhs-rhs).abs()/lhs.abs().clamp_min(1e-30))
    dense_g=g.detach().clone().requires_grad_(True)
    # Check the actual original FFT autograd adjoint on one scalar projection.
    from torch.utils.checkpoint import checkpoint
    ref_grad=torch.autograd.grad((checkpoint(original,dense_g,use_reentrant=False)*sensor).sum(),dense_g)[0]
    adjoint_gradient_error=relative(operator.adjoint(sensor),ref_grad)
    del original,dense_g,ref_grad
    torch.cuda.empty_cache()
    baseline=torch.load(BASELINE/'checkpoint_best.pt',map_location='cpu',weights_only=False)
    model=build_model(baseline['config']).to(device);model.load_state_dict(baseline['model_state']);model.eval()
    with torch.no_grad():
        output,_=experiment_forward(model,item,operator,config=baseline['config'])
        saved=torch.from_numpy(np.load(PRIORITY/'existing_inference/P09_subset_01/reconstruction.npy')).to(device)
        reproduction=relative(output.reconstruction[0,0],saved)
    del model,output
    model=build_model(config,initial=True).to(device);model.eval()
    with torch.no_grad():
        reference,_=experiment_forward(model,item,operator,config=config)
        ref=reference.reconstruction.clone();homogeneity=[]
        for c in [0.,.1,.5,2.]:
            scaled=dict(item)
            for key in ['f_var','f_var_feature','g_mean','input_mean','residual_frames']:scaled[key]=item[key]*c
            out,_=experiment_forward(model,scaled,operator,config=config)
            error=relative(out.reconstruction,ref*c) if c else float(out.reconstruction.abs().max())
            homogeneity.append({'gain':c,'relative_l2_or_zero_max':error})
        contaminated=dict(item)
        contaminated['ground_truth']=torch.randn_like(ref)
        contaminated['measured_mean']=torch.randn_like(item['input_mean'])*1e8
        contaminated['measured_variance']=torch.randn_like(item['input_mean'])*1e8
        changed,_=experiment_forward(model,contaminated,operator,config=config)
        leakage_error=relative(changed.reconstruction,ref)
    del model,reference,changed,out
    cfg3=yaml.safe_load((OUTPUT/'e3.yaml').read_text())
    model=build_model(cfg3,initial=True).to(device);model.train()
    validation=MatlabMultiVolumeDataset(DATA,'validation',cache_dir=OUTPUT/'data_cache')
    target=_to_device(validation[0],device)
    out,_=experiment_forward(model,target,operator,config=cfg3)
    terms=experiment_loss(out,target,operator,TaylorH2VarianceModel(operator),cfg3)
    mean_to_shape=torch.autograd.grad(terms.weighted_mean,out._shape,allow_unused=True,retain_graph=True)[0]
    var_to_gamma=torch.autograd.grad(terms.weighted_var,model.mean_gain_gamma,allow_unused=True,retain_graph=True)[0]
    terms.total.backward()
    finite=all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    gamma_grad=float(model.mean_gain_gamma.grad)
    shape_grads=sum(float(p.grad.square().sum()) for n,p in model.named_parameters()
                    if n!='mean_gain_gamma' and p.grad is not None)**.5
    record={'complete':True,'forward_relative_l2':forward_error,'squared_forward_relative_l2':square_error,
            'adjoint_inner_product_relative_error':adjoint_error,'adjoint_gradient_relative_l2':adjoint_gradient_error,
            'baseline_P09_subset01_reproduction_relative_l2':reproduction,
            'e1_homogeneity':homogeneity,'e1_target_gt_contamination_error':leakage_error,
            'e3_mean_gradient_to_shape_is_none':mean_to_shape is None,
            'e3_variance_gradient_to_gain_is_none':var_to_gamma is None,
            'e3_gamma_gradient':gamma_grad,'e3_network_gradient_norm':shape_grads,
            'all_gradients_finite':bool(finite),'seconds':time.monotonic()-start}
    record['passed']=(max(forward_error,square_error,adjoint_error,adjoint_gradient_error,reproduction)<1e-4 and
        max(x['relative_l2_or_zero_max'] for x in homogeneity)<1e-4 and leakage_error==0 and
        mean_to_shape is None and var_to_gamma is None and finite and shape_grads>0)
    write_json(OUTPUT/'gpu_checks.json',record);print(json.dumps(record,indent=2),flush=True)
    if not record['passed']:raise RuntimeError('GPU numerical acceptance failed')


def calibrate():
    torch.set_num_threads(4);torch.cuda.set_device(0);install_data()
    directory=OUTPUT/'illumination';directory.mkdir(parents=True,exist_ok=True)
    specs={}
    for name,start,na in [('train',2026091000,.05),('validation',2026091100,.05),
                          ('audit',2026091200,.05),('mismatch_na04479',2026091200,.04479)]:
        marker=directory/(name+'.json');path=directory/(name+'.npy')
        if marker.exists():
            spec=json.loads(marker.read_text())
            if sha256(path)!=spec['sha256']:raise ValueError('Changed illumination bank')
        else:
            started=time.monotonic();spec=generate_bank(path,list(range(start,start+16)),na=na)
            spec['sha256']=sha256(path);spec['seconds']=time.monotonic()-started
            write_json(marker,spec)
        specs[name]=spec;print(json.dumps({'bank_ready':name,'seconds':spec.get('seconds')}),flush=True)
    bank=np.load(directory/'train.npy',mmap_mode='r')
    # Streaming first/second moments avoid materializing a second 2.7 GB array.
    mean=np.zeros(bank.shape[1:],np.float64);m2=np.zeros_like(mean);n=0
    for frame in bank:
        n+=1;delta=frame-mean;mean+=delta/n;m2+=delta*(frame-mean)
    variance=m2/(n-1);sigma=float(np.sqrt(variance.mean()))
    np.savez(directory/'calibration_moments.npz',mean=mean,variance=variance,sigma=sigma)
    config=yaml.safe_load((OUTPUT/'e2.yaml').read_text())
    config['three_way']['covariance']['sigma']=sigma
    operator=load_operator(config,torch.device('cuda:0'))
    vm=CorrelatedVariance(operator,config)
    full=vm.active_bank()
    dataset=MatlabMultiVolumeDataset(DATA,'train',cache_dir=OUTPUT/'data_cache')
    rows=[];candidate_counts=[128,256,512,1024]
    for sample in ['P01','P08','P10']:
        index=next(i for i,k in enumerate(dataset.keys) if k.sample_id==sample and k.subset_index==1)
        item=_to_device(dataset[index],torch.device('cuda:0'))
        from training.multivolume_trainer import _analytic_beta0
        anchor=item['f_var']*_analytic_beta0(operator,item['f_var'],item['input_mean'])[:,None,None,None,None]
        scored={}
        for count in candidate_counts:
            vm._banks['train']=full[:count]
            g=anchor.detach().clone().requires_grad_(True);start=time.monotonic()
            pred=vm(g,item['measured_mean'])
            loss=F.smooth_l1_loss(torch.log(pred.clamp_min(0)+1e-6),torch.log(item['measured_variance']+1e-6))
            grad=torch.autograd.grad(loss,g)[0]
            torch.cuda.synchronize()
            scored[count]=(pred.detach(),grad.detach(),time.monotonic()-start)
            print(json.dumps({'sample':sample,'count':count,'seconds':scored[count][2]}),flush=True)
        for count in candidate_counts:
            p,g,seconds=scored[count];rp,rg,_=scored[1024]
            rows.append({'sample':sample,'count':count,'variance_relative_l2':relative(p,rp),
                         'gradient_relative_l2':relative(g,rg),'seconds':seconds})
    count=next(n for n in candidate_counts if all(r['variance_relative_l2']<=.05 and r['gradient_relative_l2']<=.10
                       for r in rows if r['count']==n))
    # Streamed custom backward must equal an ordinary autograd implementation.
    small=full[:4];vm._banks['train']=small
    g=anchor.detach().clone().requires_grad_(True)
    pred=vm(g,item['measured_mean']);weight=torch.linspace(.1,1,pred.numel(),device=g.device).reshape(pred.shape)
    grad=torch.autograd.grad((pred*weight).sum(),g)[0]
    g2=anchor.detach().clone().requires_grad_(True)
    y=operator(g2*((small-small.mean(0,keepdim=True))/sigma)[:,None])
    direct=y.square().sum(0,keepdim=True)/(len(small)-1)
    grad2=torch.autograd.grad((direct*weight).sum(),g2)[0]
    derivative_error=relative(grad,grad2);value_error=relative(pred,direct)
    if max(value_error,derivative_error)>1e-4:raise RuntimeError('Streamed variance derivative failed')
    config['three_way']['covariance']['train_count']=count
    (OUTPUT/'e2.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    record={'complete':True,'banks':specs,'sigma':sigma,'train_count':count,'validation_count':1024,
            'prefix_checks':rows,'streamed_value_relative_l2':value_error,'streamed_gradient_relative_l2':derivative_error,
            'unit':'g = density * fixed illumination standard deviation',
            'banks_use_no_acquisition_frames_or_gt':True,'prefix_checks_use_train_anchors_and_holdout_statistics':True,'mismatch_NA':.04479}
    write_json(OUTPUT/'calibration.json',record);print(json.dumps(record,indent=2),flush=True)
