"""Isolated experimental adapters. Original trainer/model/data files stay intact."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import datasets.matlab_multivolume_dataset as data_module
import training.multivolume_trainer as trainer
from losses.self_supervised_losses import LossBreakdown, TaylorH2VarianceModel, total_variation_3d
from train_volume import _model_from_config
from tools.three_way_physics import SparseLFM, CorrelatedVariance

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'outputs/parallel_validation_scale_cov_mean_20260907'
BASELINE=ROOT/'outputs/multivolume_n10_no_mean_run01'
DATA=ROOT/'data/speckle_data_now'
PRIORITY=ROOT/'outputs/priority_validation_20260907'
CONFIG_PATH=ROOT/'configs/three_way_checkpoint.yaml'
ORIGINAL_FORWARD=trainer._forward
ORIGINAL_LOAD_OPERATOR=trainer._load_operator
ORIGINAL_EVALUATE=trainer._evaluate
ORIGINAL_LOSS=trainer._loss


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def state_hash(state):
    digest=hashlib.sha256()
    for name,value in state.items():
        digest.update(name.encode());digest.update(str((value.dtype,tuple(value.shape))).encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False));temporary.replace(path)


def relocated_index(root):
    """Accept only the documented old-root -> new-root relocation, in memory."""
    root=Path(root).resolve()
    if root!=DATA.resolve():raise ValueError('Unexpected dataset root')
    splits=json.loads((root/'dataset_splits.json').read_text())
    final=json.loads((root/'FINAL_DATASET_MANIFEST.json').read_text())
    if not splits.get('dataset_complete') or not final.get('complete'):raise ValueError('Incomplete dataset')
    old=Path(final['dataset_root'])
    if old!=ROOT/'data/matlab_cells_pilot_v2_r04':raise ValueError('Unknown relocation provenance')
    fmap={x['sample_id']:x for x in final['samples']}
    if set(fmap)!={x['sample_id'] for x in splits['samples']}:raise ValueError('Split mismatch')
    indexed={k:[] for k in data_module.EXPECTED_SPLITS};objects={k:[] for k in indexed}
    files=[root/'dataset_splits.json',root/'FINAL_DATASET_MANIFEST.json']
    for item in splits['samples']:
        name=item['sample_id'];split=item['split'];directory=root/name
        if Path(fmap[name]['sample_dir'])!=old/name:raise ValueError('Foreign sample path')
        manifest=directory/'validation_manifest.json'
        if not json.loads(manifest.read_text()).get('complete'):raise ValueError('Unvalidated sample')
        objects[split].append(name);files.extend([manifest,directory/'prepared.mat'])
        for subset in range(1,11):
            files.append(directory/'subsets'/f'subset_{subset:02d}.mat')
            indexed[split].append(data_module.DatasetItemKey(name,subset,split,directory))
        frames=sorted((directory/'sensor_frames').glob('frame_*.mat'))
        if len(frames)!=100:raise ValueError('Expected exactly 100 frames')
        files.extend(frames)
    for split,names in objects.items():
        if tuple(names)!=data_module.EXPECTED_SPLITS[split]:raise ValueError('Changed object split')
    return indexed,data_module._source_signature(files)


def install_data():
    data_module.load_dataset_index=relocated_index


def build_model(config, *, initial=False):
    model=_model_from_config(config)
    if initial:
        shared=torch.load(config['three_way']['initial_state'],map_location='cpu',weights_only=False)
        model.load_state_dict(shared['model_state'])
        if state_hash(model.state_dict())!=shared['state_sha256']:raise ValueError('Initialization mismatch')
    if config.get('three_way',{}).get('kind')=='e3':
        model.register_parameter('mean_gain_gamma',torch.nn.Parameter(torch.zeros(())))
    return model


def load_operator(config,device, *, sparse=True):
    cache=Path(json.loads((BASELINE/'run_contract.json').read_text())['selected_psf_cache'])
    original=ORIGINAL_LOAD_OPERATOR(config,CONFIG_PATH,device,selected_h_cache=cache)
    return SparseLFM(original,OUTPUT/'sparse_operator') if sparse else original


def experiment_forward(model,item,operator,beta0=None, *, config):
    kind=config.get('three_way',{}).get('kind','baseline')
    if kind=='e1':
        reference=float(config['three_way']['brightness_reference'])
        scale=item['input_mean'].mean(dim=(1,2,3))/reference
        # The outer scale is never floored: exact black inputs must produce zero.
        divisor=torch.where(scale>0,scale,torch.ones_like(scale))
        normalized=dict(item)
        for name in ['f_var','f_var_feature','g_mean','input_mean','residual_frames']:
            normalized[name]=item[name]/divisor.reshape((-1,)+(1,)*(item[name].ndim-1))
        normalized_beta=trainer._analytic_beta0(operator,normalized['f_var'],normalized['input_mean'])
        output,_=ORIGINAL_FORWARD(model,normalized,operator,normalized_beta)
        output.reconstruction=output.reconstruction*scale[:,None,None,None,None]
        output.residual=output.residual*scale[:,None,None,None,None]
        output._input_scale=scale
        return output,normalized_beta
    output,beta0=ORIGINAL_FORWARD(model,item,operator,beta0)
    if kind=='e3':
        base=model.module if hasattr(model,'module') else model
        q=output.reconstruction/output.reconstruction.sum(dim=(1,2,3,4),keepdim=True).clamp_min(1e-30)
        with torch.no_grad():
            hq=operator(q.detach())
            a0=(hq*item['input_mean']).sum(dim=(1,2,3))/hq.square().sum(dim=(1,2,3)).clamp_min(1e-30)
            a0=a0.clamp_min(0)
        a=a0*(1+float(config['three_way']['gain_bound'])*torch.tanh(base.mean_gain_gamma))
        output.reconstruction=q*a[:,None,None,None,None]
        output._shape=q;output._gain=a;output._mean_prediction=hq*a[:,None,None,None]
    return output,beta0


def experiment_loss(output,item,operator,variance_model,config):
    kind=config.get('three_way',{}).get('kind','baseline')
    if kind!='e3':return ORIGINAL_LOSS(output,item,operator,variance_model,config)
    mu=item['measured_mean'];target=item['measured_variance'];pred_mu=output._mean_prediction
    scale=mu.detach().abs().mean().clamp_min(1e-8)
    raw_mean=F.smooth_l1_loss(pred_mu,mu)
    mean=F.smooth_l1_loss(pred_mu/scale,mu/scale)
    pred=variance_model(output._shape,mu)
    eps=float(config['loss']['var_log_eps'])
    normalized_pred=pred/pred.mean(dim=(-2,-1),keepdim=True).clamp_min(1e-30)
    normalized_target=target/target.mean(dim=(-2,-1),keepdim=True).clamp_min(1e-30)
    var=F.smooth_l1_loss(torch.log(normalized_pred.clamp_min(0)+eps),torch.log(normalized_target+eps))
    physical_pred=pred*output._gain.detach()[:,None,None,None].square()
    raw_var=F.smooth_l1_loss(physical_pred,target)
    tv=total_variation_3d(output._shape*output._gain.detach()[:,None,None,None,None],
                         z_weight=config['loss']['lambda_tv_z'])
    weighted_tv=tv*config['loss']['lambda_tv'];zero=tv.new_zeros(())
    return LossBreakdown(mean+var+weighted_tv,raw_mean,mean,mean,raw_var,var,var,zero,zero,
                         tv,weighted_tv,pred_mu,physical_pred)


def install_training(config):
    install_data()
    trainer._model_from_config=lambda c:build_model(c,initial=True)
    trainer._load_operator=lambda c,p,d,**kw:load_operator(c,d)
    trainer._forward=lambda model,item,operator,beta0=None:experiment_forward(
        model,item,operator,beta0,config=config)
    trainer._loss=experiment_loss
    if config['three_way']['kind']=='e2':
        trainer.TaylorH2VarianceModel=lambda operator,**kw:CorrelatedVariance(operator,config)
    def optimizer(model,c):
        groups=[{'params':[p for n,p in model.named_parameters() if n not in ['raw_beta','mean_gain_gamma'] and p.requires_grad],
                 'lr':float(c['optimization']['lr_network'])},
                {'params':[model.raw_beta],'lr':float(c['optimization']['lr_beta'])}]
        if hasattr(model,'mean_gain_gamma'):
            groups.append({'params':[model.mean_gain_gamma],'lr':float(c['three_way']['lr_gain'])})
        return torch.optim.Adam(groups)
    trainer._optimizer=optimizer
    def evaluate(**kwargs):
        vm=kwargs['variance_model']
        if isinstance(vm,CorrelatedVariance):vm.evaluating=True
        try:summary,rows=ORIGINAL_EVALUATE(**kwargs)
        finally:
            if isinstance(vm,CorrelatedVariance):vm.evaluating=False
        if kwargs['rank']==0:
            by_object={}
            for row in rows:
                row['selection_score']=row['total_loss']
                by_object.setdefault(row['sample_id'],[]).append(row['selection_score'])
            summary['selection_score']=float(np.mean([np.mean(x) for x in by_object.values()]))
        summary=trainer._broadcast_object(summary if kwargs['rank']==0 else None,kwargs['rank'],kwargs['world_size'])
        return summary,rows
    trainer._evaluate=evaluate


def run_train(config_path,resume=False):
    config=yaml.safe_load(Path(config_path).read_text())
    torch.set_num_threads(4)
    contract=json.loads((OUTPUT/'preflight.json').read_text())
    _,fingerprint=relocated_index(DATA)
    if fingerprint!=contract['dataset_fingerprint']:raise ValueError('Dataset changed after preflight')
    for name,digest in contract['source_sha256'].items():
        if sha256(ROOT/name)!=digest:raise ValueError(f'Source changed after preflight: {name}')
    install_training(config)
    folder=Path(config['experiment']['output_dir'])
    args=SimpleNamespace(config=str(config_path),max_steps=None,global_batch_size=None,validate_every=None,
                         phase_chunk_size=None,resume=str(folder/'checkpoint_last.pt') if resume else None,
                         output_dir=str(folder),limit_validation_items=None,limit_test_items=None)
    trainer.run_training(args)
