from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import tifffile
import yaml

from . import data
from .geometry import prepare_shifts
from .models import Baseline
from .physics import operator


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False


def device_for(cpu):
    if cpu: return torch.device('cpu')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable. Run on the GPU host; CPU checks do not certify GPU readiness.')
    free,total=torch.cuda.mem_get_info()
    if free < 10*1024**3: raise RuntimeError('Less than 10 GiB free GPU memory; choose another GPU. Existing processes are never stopped.')
    return torch.device('cuda:0')


def tensor(value,device):
    return torch.from_numpy(np.ascontiguousarray(value)).to(device)[None,None]


def losses(model,op,item,device):
    mean=tensor(item['mean'],device)
    volume=model(mean)
    if not torch.isfinite(volume).all(): raise FloatingPointError('Nonfinite reconstruction')
    if model.method=='serenet':
        mse=((op(volume)-mean)/model.input_scale).square().mean()
        negative=torch.relu(-volume/model.output_scale).square().mean()
        return mse+10*negative, mse, volume
    gt=tensor(item['truth'],device)
    mse=((volume-gt)/model.target_scale).square().mean()
    return mse,mse,volume


@torch.no_grad()
def validate(model,op,device,limit=0):
    model.eval(); by_obj={}
    records=data.keys('validation')
    if limit: records=records[:limit]
    for obj,sub in records:
        item=data.sample(obj,sub,supervised=model.method=='vcdnet')
        _,mse,_=losses(model,op,item,device)
        by_obj.setdefault(obj,[]).append(float(mse))
    model.train()
    return float(np.mean([np.mean(v) for v in by_obj.values()]))


def checkpoint(path,model,opt,state,config,contract):
    payload={'model':model.state_dict(),'optimizer':opt.state_dict(),'state':state.copy(),
             'config':config,'contract':contract,'contract_sha256':data.digest(data.BASE/'manifests/data.json'),
             'source_sha256':data.digest(data.BASE/'manifests/sources.json'),
             'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
                    'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
    tmp=path.with_suffix('.tmp'); torch.save(payload,tmp); tmp.replace(path)


def train(args):
    config=yaml.safe_load((data.BASE/'configs'/f'{args.method}.yaml').read_text())
    if args.seed is not None: config['seed']=args.seed
    if args.epochs: config['epochs']=args.epochs
    if args.lr: config['learning_rate']=args.lr
    config['fixed_sample']=args.fixed_sample
    if config['batch_size']!=1 or config['precision']!='float32' or config['angles']!=2401 or config['depths']!=10:
        raise ValueError('This adapter requires batch1, FP32, 2401 angles and 10 depth planes')
    if config['epochs']<1 or config['validate_every']<1 or config['learning_rate']<=0 or args.max_steps<0:
        raise ValueError('Invalid training schedule')
    device=device_for(args.cpu); seed_all(config['seed']); info=data.contract()
    model=Baseline(args.method,info).to(device)
    op=operator(device) if args.method=='serenet' else None
    opt=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    out=Path(args.output) if args.output else data.BASE/'outputs'/f'{args.method}_seed{config["seed"]}'
    out=out.resolve()
    if (out/'checkpoint_last.pt').exists() and not args.resume:
        raise FileExistsError('Output already has a checkpoint; use --resume or a new --output')
    out.mkdir(parents=True,exist_ok=True)
    state={'epoch':0,'cursor':0,'order':[],'step':0,'best':None}
    if args.resume:
        saved=torch.load(args.resume,map_location=device,weights_only=False)
        if saved['config']['method']!=args.method or saved['contract_sha256']!=data.digest(data.BASE/'manifests/data.json'):
            raise ValueError('Resume method/data mismatch')
        if saved['source_sha256']!=data.digest(data.BASE/'manifests/sources.json'):
            raise ValueError('Upstream source changed')
        for key in ('seed','learning_rate','fixed_sample'):
            if saved['config'].get(key)!=config.get(key): raise ValueError(f'Resume {key} mismatch')
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer']); state=saved['state']
        rng=saved['rng']; random.setstate(rng['python']); np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'].cpu())
        if rng['cuda'] is not None and device.type=='cuda': torch.cuda.set_rng_state_all([x.cpu() for x in rng['cuda']])
        def equal_state(a,b):
            if torch.is_tensor(a): return torch.equal(a.detach().cpu(),b.detach().cpu())
            if isinstance(a,dict): return a.keys()==b.keys() and all(equal_state(a[k],b[k]) for k in a)
            if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal_state(x,y) for x,y in zip(a,b))
            if isinstance(a,np.ndarray): return np.array_equal(a,b)
            return a==b
        restored={'model_exact':equal_state(model.state_dict(),saved['model']),
                  'optimizer_exact':equal_state(opt.state_dict(),saved['optimizer']),
                  'python_rng_exact':equal_state(random.getstate(),rng['python']),
                  'numpy_rng_exact':equal_state(np.random.get_state(),rng['numpy']),
                  'torch_rng_exact':equal_state(torch.get_rng_state(),rng['torch']),
                  'cuda_rng_exact':equal_state(torch.cuda.get_rng_state_all(),rng['cuda']) if device.type=='cuda' else None,
                  'start_step':state['step'],'cursor':state['cursor'],'epoch':state['epoch']}
        if not all(v for k,v in restored.items() if k.endswith('_exact') and v is not None):
            raise RuntimeError('Checkpoint state was not restored exactly')
        data.write_json(out/'resume_restore.json',restored)
    data.write_json(out/'config.json',config)
    train_keys=data.keys('train')
    if args.fixed_sample: train_keys=[train_keys[0]]
    model.train(); initial_step=state['step']; values=[]
    before=next(model.parameters()).detach().clone()
    start=time.perf_counter()
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    while state['epoch']<config['epochs']:
        if not state['order']:
            state['order']=np.random.permutation(len(train_keys)).tolist(); state['cursor']=0
        while state['cursor']<len(state['order']):
            obj,sub=train_keys[state['order'][state['cursor']]]
            item=data.sample(obj,sub,supervised=args.method=='vcdnet')
            opt.zero_grad(set_to_none=True)
            loss,mse,_=losses(model,op,item,device)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
            loss.backward()
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all(): raise FloatingPointError('Nonfinite gradient')
            opt.step(); state['step']+=1; state['cursor']+=1
            row={'step':state['step'],'epoch':state['epoch']+1,'object':obj,'subset':sub,'loss':float(loss),'mse':float(mse)}
            values.append(row)
            with open(out/'training.jsonl','a') as f: f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
            if args.max_steps and state['step']-initial_step>=args.max_steps: break
        if state['cursor']==len(state['order']):
            state['epoch']+=1; state['cursor']=0; state['order']=[]
        stopping=bool(args.max_steps and state['step']-initial_step>=args.max_steps)
        if stopping or (not state['order'] and state['epoch']%config['validate_every']==0) or state['epoch']==config['epochs']:
            val=validate(model,op,device,args.validation_limit)
            if state['best'] is None or val<state['best']:
                state['best']=val
                checkpoint(out/'checkpoint_best.pt',model,opt,state,config,info)
            data.write_json(out/'validation.json',{'step':state['step'],'loss':val,'limit':args.validation_limit,'criterion':config['selection']})
        if stopping or (not state['order'] and state['epoch']%config['validate_every']==0) or state['epoch']==config['epochs']:
            checkpoint(out/'checkpoint_last.pt',model,opt,state,config,info)
        if stopping: break
    if device.type=='cuda': torch.cuda.synchronize()
    changed=bool(not torch.equal(before,next(model.parameters()).detach()))
    data.write_json(out/'run_summary.json',{'device':str(device),'steps_this_run':state['step']-initial_step,
        'total_steps':state['step'],'epoch':state['epoch'],'parameters_changed':changed,
        'first_loss':values[0]['loss'] if values else None,'last_loss':values[-1]['loss'] if values else None,
        'seconds':time.perf_counter()-start,'peak_memory_gib':torch.cuda.max_memory_allocated()/1024**3 if device.type=='cuda' else 0.,
        'resumed':bool(args.resume),'fixed_sample':args.fixed_sample})
    if values and not changed: raise RuntimeError('Parameters did not update')


def load_trained(args):
    device=device_for(args.cpu)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if saved['config']['method']!=args.method: raise ValueError('Checkpoint method mismatch')
    if saved['contract_sha256']!=data.digest(data.BASE/'manifests/data.json'): raise ValueError('Checkpoint data mismatch')
    if saved['source_sha256']!=data.digest(data.BASE/'manifests/sources.json'): raise ValueError('Source mismatch')
    seed_all(saved['config']['seed'])
    model=Baseline(args.method,saved['contract']).to(device)
    model.load_state_dict(saved['model']); model.eval()
    return model,device,saved


def sync(device):
    if device.type=='cuda': torch.cuda.synchronize()


@torch.no_grad()
def infer(args):
    model,device,saved=load_trained(args)
    out=Path(args.output) if args.output else Path(args.checkpoint).parent/'inference'
    out.mkdir(parents=True,exist_ok=True)
    records=[(f'real_{args.field}',args.subset)] if args.field else data.keys(args.split)
    if args.object: records=[x for x in records if x[0]==args.object and x[1]==args.subset]
    if args.limit: records=records[:args.limit]
    if not records: raise ValueError('No samples selected')
    rows=[]
    for obj,sub in records:
        sync(device); start=time.perf_counter()
        item=data.real_sample(args.field,sub) if args.field else data.sample(obj,sub)
        x=tensor(item['mean'],device); sync(device); prep=time.perf_counter()-start
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
        begin=time.perf_counter(); pred=model(x).clamp_min(0); sync(device); network=time.perf_counter()-begin
        volume=pred[0,0].cpu().numpy()
        if not np.isfinite(volume).all() or volume.shape!=(10,*item['mean'].shape): raise ValueError('Invalid output')
        target=out/f'{obj}_{sub:02d}.tif'
        tifffile.imwrite(target,volume.astype(np.float32),metadata={'axes':'ZYX','z_um':data.Z,'pixel_pitch_um':220/49/4})
        sync(device)
        row={'object':obj,'subset':sub,'path':str(target.resolve()),'shape':list(volume.shape),'preprocess_seconds':prep,
             'network_seconds':network,'total_seconds':time.perf_counter()-start,
             'peak_memory_gib':torch.cuda.max_memory_allocated()/1024**3 if device.type=='cuda' else 0.,
             'checkpoint':str(Path(args.checkpoint).resolve()),'method':args.method,'device':str(device),'intensity':'physical linear; negative values clipped'}
        data.write_json(target.with_suffix('.json'),row); rows.append(row)
        print(json.dumps(row),flush=True)
    data.write_json(out/'inference_summary.json',rows)


def evaluate(args):
    from skimage.metrics import structural_similarity
    folder=Path(args.predictions); rows=[]
    for path in sorted(folder.glob('*.tif')):
        obj,subset=path.stem.rsplit('_',1)
        if obj not in SPLIT_EVAL: continue
        pred=tifffile.imread(path); gt=data.truth(obj)
        if pred.shape!=gt.shape: raise ValueError('Prediction geometry mismatch')
        scale=data.contract()['vcd_target_scale']
        delta=(pred.astype(np.float64)-gt)/scale
        mse=float(np.mean(delta**2)); mae=float(np.mean(abs(delta)))
        ssim=float(np.mean([structural_similarity(gt[z],pred[z],data_range=scale) for z in range(10)]))
        rows.append({'object':obj,'subset':int(subset),'normalized_mse':mse,'normalized_mae':mae,
                     'psnr_db':float(-10*np.log10(max(mse,1e-30))),'ssim_slice_mean':ssim})
    if not rows: raise ValueError('No validation/test predictions')
    columns=['normalized_mse','normalized_mae','psnr_db','ssim_slice_mean']
    by={obj:{k:float(np.mean([r[k] for r in rows if r['object']==obj])) for k in columns} for obj in sorted({r['object'] for r in rows})}
    data.write_json(folder/'metrics.json',{'rows':rows,'objects':by,'object_macro':{k:float(np.mean([v[k] for v in by.values()])) for k in columns},
        'normalization':'fixed training GT maximum * 1.05; no test-dependent fit','ssim':'mean of 10 lateral slices, not a volumetric SSIM',
        'test_complete':all((obj,s) in {(r['object'],r['subset']) for r in rows} for obj,s in data.keys('test'))})


SPLIT_EVAL=sum([data.SPLITS['validation'],data.SPLITS['test']],[])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','train','infer','evaluate'])
    parser.add_argument('--method',choices=['serenet','vcdnet'],default='serenet')
    parser.add_argument('--gpu',default='3'); parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--output'); parser.add_argument('--seed',type=int); parser.add_argument('--epochs',type=int)
    parser.add_argument('--lr',type=float); parser.add_argument('--max-steps',type=int,default=0)
    parser.add_argument('--validation-limit',type=int,default=0); parser.add_argument('--fixed-sample',action='store_true')
    parser.add_argument('--resume'); parser.add_argument('--checkpoint')
    parser.add_argument('--split',choices=['validation','test'],default='test')
    parser.add_argument('--object'); parser.add_argument('--subset',type=int,default=1)
    parser.add_argument('--field',choices=['45','55']); parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--predictions')
    args=parser.parse_args(); os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    torch.set_num_threads(4)
    if args.command=='prepare': data.prepare(); prepare_shifts()
    elif args.command=='train': train(args)
    elif args.command=='infer': infer(args)
    else: evaluate(args)


if __name__=='__main__': main()
