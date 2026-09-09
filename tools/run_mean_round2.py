"""Controlled second round: E3 continuation, mean ablation and bounded mean gradients."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from tools.three_way_experiment import OUTPUT as PREVIOUS,DATA,BASELINE,sha256,write_json,relocated_index
OUTPUT=ROOT/'outputs/mean_refinement_round2_20260908'
ARMS=['r0_continue','r1_no_mean','r2_input_scale','r3_mean_shape_005','r4_mean_shape_010']
GPU_POLICY={'training':'sequential arms; all idle A40 GPUs in requested pool, at most global batch 8',
            'resume':'preserve checkpoint world_size; fail if too few idle GPUs',
            'evaluation':'one task per idle A40, in batches','global_batch_size':8}

def prepare(resume=False,gpu_request='auto'):
    import torch
    import yaml
    from tools.mean_round2_experiment import build_model
    from tools.three_way_experiment import state_hash
    marker=OUTPUT/'preflight.json'
    if marker.exists():
        if not resume:raise FileExistsError('Use --resume for this existing round')
        record=json.loads(marker.read_text())
        for path,digest in record['source_hashes'].items():
            if sha256(path)!=digest:raise ValueError(f'Frozen training source changed: {path}')
        for path,digest in record['config_hashes'].items():
            if sha256(path)!=digest:raise ValueError(f'Frozen configuration changed: {path}')
        if relocated_index(DATA)[1]!=record['dataset_fingerprint']:raise ValueError('Dataset fingerprint changed')
        if sha256(record['source_checkpoint'])!=record['source_checkpoint_sha256']:raise ValueError('Source E3 weights changed')
        return record
    OUTPUT.mkdir(parents=True,exist_ok=True)
    previous=json.loads((PREVIOUS/'preflight.json').read_text())
    for path,digest in previous['source_sha256'].items():
        if sha256(ROOT/path)!=digest:raise ValueError('First-round frozen source changed: '+path)
    indexed,fingerprint=relocated_index(DATA)
    if fingerprint!=previous['dataset_fingerprint']:raise ValueError('Dataset changed since round one')
    for path,digest in previous['dataset_content_sha256'].items():
        if sha256(DATA/path)!=digest:raise ValueError('Dataset content changed: '+path)
    source=PREVIOUS/'e3/checkpoint_last.pt';source_hash=sha256(source)
    checkpoint=torch.load(source,map_location='cpu',weights_only=False)
    if checkpoint['completed_steps']!=200:raise ValueError('Expected actual E3 final step 200')
    source_gamma=float(checkpoint['model_state']['mean_gain_gamma'])
    initial_hashes={};configs={}
    for name in ARMS:
        cfg=copy.deepcopy(checkpoint['config'])
        cfg['experiment'].update(name='mean_round2_'+name,seed=20260901,output_dir=str(OUTPUT/name))
        cfg['data'].update(root=str(DATA),cache_dir=str(PREVIOUS/'data_cache'),precompute_cache=False)
        cfg['optimization'].update(max_steps=200,global_batch_size=8,micro_batch_per_gpu=1,
                lr_network=1e-4,lr_beta=1e-5,validate_every=20,checkpoint_every=20)
        cfg['three_way']['lr_gain']=1e-5
        cfg['loss']['lambda_mean']=1.0
        cfg['round2']={'kind':name,'name':name,'source_checkpoint':str(source),
             'source_checkpoint_sha256':source_hash,'start_step':200,
             'numerical_precision':'fp32_tf32_disabled',
             'shape_gradient_budget':.05 if name=='r3_mean_shape_005' else .1 if name=='r4_mean_shape_010' else 0.,
             'ramp_steps':50,'shape_coefficient_cap':1.0,'normalize_input':name=='r2_input_scale',
             'mean_gain_trainable':name!='r1_no_mean','initial_gamma':0.,
             'reset_optimizer':True,'selection_rule':'common original E3 validation mean + normalized variance + TV, object macro; no GT',
             'illumination_NA':.05,'detection_NA':.15,'real_system_NA_status':'deferred by user',
             'gradient_limit_location':'normalized reconstructed volume q, not Adam parameter update',
             'shape_mean_definition':'SmoothL1(H(q)/mean(H(q)), mu90/mean(mu90))'}
        config_path=OUTPUT/(name+'.yaml');config_path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        initial=build_model(cfg,initial=True)
        initial_hashes[name]=state_hash(initial.state_dict());del initial
        configs[str(config_path)]=sha256(config_path)
    if len(set(initial_hashes.values()))!=1:raise ValueError('Arms do not share exact initial model state')
    sources=[ROOT/path for path in previous['source_sha256']]
    sources.extend([ROOT/'tools/mean_round2_experiment.py',ROOT/'tools/mean_round2_checks.py',Path(__file__).resolve()])
    source_hashes={str(path):sha256(path) for path in sources}
    for path in sources:
        dest=OUTPUT/'source_snapshot'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    record={'complete':True,'round':2,'dataset_fingerprint':fingerprint,
            'dataset_files_verified':len(previous['dataset_content_sha256']),
            'source_checkpoint':str(source),'source_checkpoint_sha256':source_hash,
            'source_gamma':source_gamma,'all_arms_gamma_reset':0.,'initial_state_hashes':initial_hashes,
            'source_hashes':source_hashes,'config_hashes':configs,
            'gpu_policy':GPU_POLICY,'gpu_request_at_preflight':gpu_request,
            'numerical_precision':'fp32_tf32_disabled',
            'additional_steps':200,'start_step':200,'final_total_steps':400,'global_batch':8,
            'previous_preflight_sha256':sha256(PREVIOUS/'preflight.json'),
            'old_baseline_best_sha256':sha256(BASELINE/'checkpoint_best.pt'),
            'old_baseline_final_sha256':sha256(BASELINE/'checkpoint_last.pt'),
            'psf_sha256':sha256(checkpoint['config']['psf']['H_path']),
            'no_training_gt':True,'common_validation_selection':True,'prepared_unix':time.time()}
    write_json(marker,record);print(json.dumps({'preflight_prepared':True,'arms':ARMS,'initial_state_equal':True}),flush=True)
    return record

def inventory():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,utilization.gpu',
                                  '--format=csv,noheader,nounits'],text=True)
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                     '--format=csv,noheader,nounits'],text=True)
    busy={line.split(',')[0].strip() for line in occupied.splitlines() if line.strip()}
    result={}
    for line in rows.splitlines():
        index,uuid,name,memory,utilization=[field.strip() for field in line.split(',')]
        result[int(index)]={'uuid':uuid,'name':name,'memory':int(memory),
                            'utilization':int(utilization),'busy':uuid in busy}
    return result


def parse_gpus(value):
    if value.strip().lower()=='auto':return None
    fields=value.split(',')
    if not fields or any(not field.strip().isdigit() for field in fields):
        raise ValueError('--gpus must be auto or a comma-separated list of GPU indices')
    indices=[int(field.strip()) for field in fields]
    if len(indices)!=len(set(indices)):raise ValueError('--gpus must not repeat an index')
    return indices


def _gpu_pool(inv,gpu_request):
    requested=parse_gpus(gpu_request)
    if requested is None:return sorted(gpu for gpu,info in inv.items() if 'A40' in info['name'].upper())
    for gpu in requested:
        if gpu not in inv:raise ValueError(f'Requested GPU {gpu} does not exist')
        if 'A40' not in inv[gpu]['name'].upper():raise ValueError(f'Requested GPU {gpu} is not an A40: {inv[gpu]["name"]}')
    return requested


def free_gpus(gpu_request='auto',required=None):
    inv=inventory();pool=_gpu_pool(inv,gpu_request)
    # A just-finished worker can leave a stale rolling utilization sample.
    # Never wait away a live compute process or occupied memory.
    for _ in range(10):
        if not any(not inv[g]['busy'] and inv[g]['memory']<=1024 and inv[g]['utilization']>10 for g in pool):break
        time.sleep(1);inv=inventory();pool=_gpu_pool(inv,gpu_request)
    available=[g for g in pool if not inv[g]['busy'] and inv[g]['memory']<=1024 and inv[g]['utilization']<=10]
    needed=1 if required is None else required
    if len(available)<needed:
        raise RuntimeError(f'Insufficient idle A40 GPUs: need {needed}, available {available}; '
                           f'requested pool {pool}. Occupied cards will not be shared.')
    return available if required is None else available[:required]


def training_gpus(name,gpu_request,resume=False):
    checkpoint=OUTPUT/name/'checkpoint_last.pt'
    if checkpoint.exists():
        if not resume:raise FileExistsError(f'Existing checkpoint requires --resume: {checkpoint}')
        import torch
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        world_size=payload.get('world_size')
        if not isinstance(world_size,int) or isinstance(world_size,bool) or world_size<1:
            raise ValueError(f'Checkpoint has no valid world_size: {checkpoint}')
        if int(payload['config']['optimization']['global_batch_size'])!=8:
            raise ValueError(f'Checkpoint global batch size is not 8: {checkpoint}')
        del payload
        if world_size>8:raise ValueError('Checkpoint world_size exceeds fixed global batch size 8')
        return free_gpus(gpu_request,required=world_size)
    return free_gpus(gpu_request)[:8]


def launch(worker,gpus,name=None,resume=False):
    if not gpus:raise ValueError('A worker requires at least one GPU')
    inv=inventory()
    for gpu in gpus:
        if gpu not in inv or 'A40' not in inv[gpu]['name'].upper():raise ValueError(f'GPU {gpu} is not an available A40 device')
    for _ in range(10):
        if not any(not inv[g]['busy'] and inv[g]['memory']<=1024 and inv[g]['utilization']>10 for g in gpus):break
        time.sleep(1);inv=inventory()
    for gpu in gpus:
        if inv[gpu]['busy'] or inv[gpu]['memory']>1024 or inv[gpu]['utilization']>10:
            raise RuntimeError(f'GPU {gpu} occupied; refusing shared use: {inv[gpu]}')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=','.join(inv[g]['uuid'] for g in gpus),
        SPECKLE_PHYSICAL_GPUS=','.join(map(str,gpus)),CUDA_DEVICE_ORDER='PCI_BUS_ID',
        PYTHONPATH=str(ROOT)+os.pathsep+str(ROOT/'tools'),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
        MKL_NUM_THREADS='4',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',MPLCONFIGDIR=str(OUTPUT/'mpl_cache'))
    args=[str(Path(__file__).resolve()),'--worker',worker]
    if name:args+=['--experiment',name]
    if resume:args+=['--resume']
    command=[sys.executable,*args]
    if worker=='train' and len(gpus)>1:command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={len(gpus)}',*args]
    log=OUTPUT/'logs'/f'{worker}_{name or "all"}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as handle:proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
    path=OUTPUT/'jobs'/f'{worker}_{name or "all"}.json'
    write_json(path,{'pid':proc.pid,'worker':worker,'experiment':name,'gpus':gpus,'command':command,
                    'started_unix':time.time(),'status':'running','log':str(log)})
    proc.record_path=path
    print(json.dumps({'started':worker,'experiment':name,'pid':proc.pid,'gpus':gpus,'log':str(log)}),flush=True)
    return proc

def wait(proc):
    code=proc.wait();record=json.loads(proc.record_path.read_text());record.update(
           exit_code=code,status='complete' if code==0 else 'failed',finished_unix=time.time())
    write_json(proc.record_path,record)
    if code:raise RuntimeError(f'Worker failed: {record["log"]}')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','preflight','train','evaluate','report'],default='all')
    parser.add_argument('--experiment',choices=['all',*ARMS],default='all')
    parser.add_argument('--gpus',default='auto',help='auto or comma-separated A40 index pool; occupied cards are excluded')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--worker',choices=['train','gpu-check','evaluate'])
    args=parser.parse_args()
    try:parse_gpus(args.gpus)
    except ValueError as error:parser.error(str(error))
    selected=ARMS if args.experiment=='all' else [args.experiment]
    if args.dry_run:
        print(json.dumps({'output':str(OUTPUT),'arms':selected,'gpu_request':args.gpus,'gpu_policy':GPU_POLICY,
           'gpu_check':'first idle A40 in requested pool','dry_run_no_changes':True,
           'numerical_precision':'fp32_tf32_disabled',
           'start':'E3 actual final 200; gamma reset zero',
           'additional_steps':200,'final_total_steps':400,'lr_network':1e-4,'lr_beta_gamma':1e-5,
           'shape_mean_gradient_budgets':[0,.05,.1],'checkpoint_selection':'same E3 validation objective, no GT'},indent=2));return
    if args.worker:
        if args.worker=='train':
            from tools.mean_round2_experiment import run_train
            run_train(OUTPUT/(args.experiment+'.yaml'),args.resume)
        elif args.worker=='gpu-check':
            from tools.mean_round2_checks import gpu_checks
            gpu_checks()
        else:
            from tools.mean_round2_evaluation import evaluate
            evaluate(args.experiment,resume=args.resume)
        return
    if args.stage in ['all','preflight','train']:prepare(args.resume,args.gpus)
    if args.stage in ['all','preflight','train'] and not (OUTPUT/'gpu_checks.json').exists():
        wait(launch('gpu-check',free_gpus(args.gpus,required=1)))
    if args.stage=='preflight':return
    if args.stage in ['all','train']:
        if not json.loads((OUTPUT/'gpu_checks.json').read_text()).get('passed'):raise RuntimeError('Numerical checks failed')
        for name in selected:
            if args.resume and (OUTPUT/name/'training_complete.json').exists():continue
            gpus=training_gpus(name,args.gpus,args.resume)
            print(json.dumps({'training_resource_selection':name,'gpus':gpus,
                              'world_size':len(gpus),'global_batch_size':8}),flush=True)
            wait(launch('train',gpus,name,resume=args.resume and (OUTPUT/name/'checkpoint_last.pt').exists()))
    if args.stage in ['all','evaluate']:
        pending=[name for name in selected if not (args.resume and (OUTPUT/'evaluation'/name/'complete.json').exists())]
        while pending:
            available=free_gpus(args.gpus)
            batch=pending[:len(available)];pending=pending[len(batch):]
            jobs=[];error=None
            try:
                for gpu,name in zip(available,batch):jobs.append(launch('evaluate',[gpu],name,args.resume))
            except Exception as exc:error=exc
            for proc in jobs:
                try:wait(proc)
                except Exception as exc:
                    if error is None:error=exc
            if error is not None:raise error
    if args.stage in ['all','report']:
        from tools.mean_round2_evaluation import report
        report()

if __name__=='__main__':main()
