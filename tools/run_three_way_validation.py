"""Resumeable independent E1/E2/E3 training, best/final comparison and reporting."""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from tools.three_way_experiment import OUTPUT,BASELINE,DATA,PRIORITY,sha256,write_json,state_hash


def cpu_preflight(resume):
    import h5py
    import numpy as np
    import torch
    import yaml
    from tools.three_way_experiment import relocated_index,install_data,build_model
    from datasets.matlab_multivolume_dataset import MatlabMultiVolumeDataset
    marker=OUTPUT/'preflight.json'
    if marker.exists():
        if not resume:raise FileExistsError('Existing experiment: use --resume')
        record=json.loads(marker.read_text())
        for path,digest in record['source_sha256'].items():
            if sha256(ROOT/path)!=digest:raise ValueError(f'Changed source: {path}')
        if relocated_index(DATA)[1]!=record['dataset_fingerprint']:raise ValueError('Changed dataset')
        return record
    OUTPUT.mkdir(parents=True,exist_ok=True)
    baseline=torch.load(BASELINE/'checkpoint_best.pt',map_location='cpu',weights_only=False)
    if baseline['completed_steps']!=160:raise ValueError('Baseline best must be step 160')
    if sha256(BASELINE/'checkpoint_best.pt')!='89fc1a08f901ca36143ca904e1550be5a9955f356b14c019366958879c532872':
        raise ValueError('Baseline checkpoint changed')
    final=torch.load(BASELINE/'checkpoint_last.pt',map_location='cpu',weights_only=False)
    if final['completed_steps']!=200:raise ValueError('Baseline final must be step 200')
    install_data();indexed,fingerprint=relocated_index(DATA)
    # Verify every raw sensor frame, prepared volume and RL3 subset against its
    # original MATLAB validation manifest, not just path/mtime fingerprints.
    content={};object_means={}
    for sample in json.loads((DATA/'dataset_splits.json').read_text())['samples']:
        name=sample['sample_id'];directory=DATA/name
        entries=json.loads((directory/'validation_manifest.json').read_text())['artifacts']
        wanted={'prepared.mat'}|{f'subsets/subset_{i:02d}.mat' for i in range(1,11)}|{
            f'sensor_frames/frame_{i:03d}.mat' for i in range(1,101)}
        found={}
        for entry in entries:
            rel=entry['relative_path']
            if rel in wanted:
                actual=sha256(directory/rel)
                if actual!=entry['sha256']:raise ValueError(f'Content changed: {name}/{rel}')
                found[rel]=actual;content[f'{name}/{rel}']=actual
        if set(found)!=wanted:raise ValueError(f'Missing provenance: {name}')
        if sample['split']=='train':
            means=[]
            for subset in range(1,11):
                with h5py.File(directory/'subsets'/f'subset_{subset:02d}.mat','r') as f:
                    means.append(float(np.asarray(f['input_physics_mean_float']).mean(dtype=np.float64)))
            object_means[name]=float(np.mean(means))
        print(json.dumps({'verified_object':name,'files':len(found)}),flush=True)
    reference=float(np.median(list(object_means.values())))
    config=copy.deepcopy(baseline['config'])
    config['data']['root']=str(DATA);config['data']['cache_dir']=str(OUTPUT/'data_cache')
    config['data']['var_feature_representation']='sqrt';config['data']['precompute_cache']=False
    config['psf']['H_path']=str(ROOT/config['psf']['H_path'])
    config['runtime']['psf_cache_dir']=str(ROOT/'data/.cache/psf')
    config['runtime']['tensorboard']=True
    torch.manual_seed(config['experiment']['seed'])
    initial=build_model(config)
    shared={'model_state':initial.state_dict(),'state_sha256':state_hash(initial.state_dict()),
            'seed':config['experiment']['seed']}
    torch.save(shared,OUTPUT/'initial_state.pt')
    for name in ['e1','e2','e3']:
        candidate=copy.deepcopy(config)
        candidate['experiment']['name']='three_way_'+name
        candidate['experiment']['output_dir']=str(OUTPUT/name)
        candidate['three_way']={'kind':name,'initial_state':str(OUTPUT/'initial_state.pt'),
              'initial_state_sha256':shared['state_sha256'],'brightness_reference':reference,
              'gain_bound':.2,'lr_gain':1e-4,'mean_weight':1. if name=='e3' else 0.,
              'selection_rule':'object macro native validation total loss; never GT',
              'illumination_NA':.05,'detection_NA':.15,'real_system_NA_status':'deferred by user',
              'covariance':{'train_bank':str(OUTPUT/'illumination/train.npy'),
                            'validation_bank':str(OUTPUT/'illumination/validation.npy'),
                            'audit_bank':str(OUTPUT/'illumination/audit.npy'),
                            'mismatch_bank':str(OUTPUT/'illumination/mismatch_na04479.npy'),
                            'train_count':1024,'chunk':8,'sigma':None}}
        (OUTPUT/f'{name}.yaml').write_text(yaml.safe_dump(candidate,sort_keys=False))
    sources=[]
    for directory in ['models','training','datasets','losses','physics','utils']:
        sources.extend((ROOT/directory).glob('*.py'))
    sources.extend([ROOT/'train_volume.py',ROOT/'tools/three_way_experiment.py',ROOT/'tools/three_way_physics.py'])
    source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in sources}
    for p in sources:
        dest=OUTPUT/'source_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(p.read_bytes())
    record={'complete':True,'dataset_fingerprint':fingerprint,'dataset_content_sha256':content,
            'source_sha256':source_hashes,'baseline_best_sha256':sha256(BASELINE/'checkpoint_best.pt'),
            'baseline_final_sha256':sha256(BASELINE/'checkpoint_last.pt'),
            'psf_sha256':sha256(config['psf']['H_path']),'initial_state_sha256':shared['state_sha256'],
            'brightness_reference':reference,'train_object_input_means':object_means,
            'splits':{k:len(v) for k,v in indexed.items()},'global_batch':8,'steps':200,
            'gpus':{'e1':[1],'e2':[2,3],'e3':[4,5]},'no_training_gt':True,
            'real_system_illumination_NA_not_confirmed':True}
    write_json(marker,record)
    for split in ['train','validation','test']:
        MatlabMultiVolumeDataset(DATA,split,cache_dir=OUTPUT/'data_cache').precompute()
    return record


def inventory():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
                                  '--format=csv,noheader,nounits'],text=True)
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                     '--format=csv,noheader,nounits'],text=True)
    busy={s.split(',')[0].strip() for s in occupied.splitlines() if s.strip()}
    return {int(a):{'uuid':b.strip(),'memory':int(c),'utilization':int(d),'busy':b.strip() in busy}
            for a,b,c,d in [line.split(',') for line in rows.splitlines()]}


def launch(worker,gpus,*,experiment=None,resume=False,extra=()):
    inv=inventory()
    # Utilization can retain our completed worker's last rolling sample.
    for attempt in range(10):
        if not any(not inv[g]['busy'] and inv[g]['memory']<=1024 and inv[g]['utilization']>10 for g in gpus):break
        time.sleep(1);inv=inventory()
    for gpu in gpus:
        if inv[gpu]['busy'] or inv[gpu]['memory']>1024 or inv[gpu]['utilization']>10:
            raise RuntimeError(f'Refusing occupied GPU {gpu}: {inv[gpu]}')
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=','.join(inv[x]['uuid'] for x in gpus)
    env['SPECKLE_PHYSICAL_GPUS']=','.join(map(str,gpus));env['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    env['PYTHONPATH']=str(ROOT)+os.pathsep+str(ROOT/'tools')
    env['OMP_NUM_THREADS']='4';env['OPENBLAS_NUM_THREADS']='4';env['MKL_NUM_THREADS']='4'
    env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    env['MPLCONFIGDIR']=str(OUTPUT/'mpl_cache')
    args=[str(Path(__file__).resolve()),'--worker',worker,*extra]
    if experiment:args+=['--experiment',experiment]
    if resume:args+=['--resume']
    command=[sys.executable,*args]
    if worker=='train' and len(gpus)>1:
        command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={len(gpus)}',*args]
    logs=OUTPUT/'logs';logs.mkdir(parents=True,exist_ok=True)
    label=experiment or ('sample_'+str(extra[list(extra).index('--brightness-sample')+1]) if '--brightness-sample' in extra else 'all')
    path=logs/f'{worker}_{label}.log'
    handle=path.open('a')
    proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
    handle.close()
    write_json(OUTPUT/'jobs'/f'{worker}_{label}.json',
               {'pid':proc.pid,'command':command,'gpus':gpus,'started_unix':time.time(),'log':str(path)})
    print(json.dumps({'started':worker,'experiment':experiment,'pid':proc.pid,'gpus':gpus,'log':str(path)}),flush=True)
    return proc


def wait(proc):
    result=proc.wait()
    if result:raise RuntimeError(f'Worker PID {proc.pid} failed with exit {result}; inspect logs')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','preflight','calibrate','train','evaluate','report'],default='all')
    parser.add_argument('--experiment',choices=['all','e1','e2','e3','baseline'],default='all')
    parser.add_argument('--gpus',default='e1=1;e2=2,3;e3=4,5')
    parser.add_argument('--brightness-sample',type=int,choices=[0,1,2])
    parser.add_argument('--resume',action='store_true');parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--worker',choices=['train','gpu-check','calibrate','evaluate','brightness'])
    args=parser.parse_args()
    if args.dry_run:
        print(json.dumps({'output':str(OUTPUT),'stage':args.stage,'gpus':args.gpus,'steps':200,
             'global_batch':8,'cold_start':True,'compare':['baseline','e1','e2','e3'],
             'checkpoints':['best','final'],'illumination_NA':.05,'PSF_NA':.15},indent=2));return
    if args.worker:
        if args.worker=='train':
            from tools.three_way_experiment import run_train
            run_train(OUTPUT/f'{args.experiment}.yaml',args.resume)
        elif args.worker in ['gpu-check','calibrate']:
            from tools.three_way_checks import gpu_checks,calibrate
            (gpu_checks if args.worker=='gpu-check' else calibrate)()
        elif args.worker=='brightness':
            from tools.three_way_report import brightness_inputs
            brightness_inputs(args.brightness_sample)
        else:
            from tools.three_way_report import evaluate
            evaluate(args.experiment,resume=args.resume)
        return
    mapping={name:[int(v) for v in values.split(',')] for name,values in
             (part.split('=') for part in args.gpus.split(';'))}
    selected=['e1','e2','e3'] if args.experiment=='all' else [args.experiment]
    if args.stage in ['preflight','all']:cpu_preflight(args.resume)
    if args.stage in ['preflight','all'] and not (OUTPUT/'gpu_checks.json').exists():
        wait(launch('gpu-check',mapping['e1']))
    if args.stage=='preflight':return
    jobs={}
    if args.stage in ['train','all']:
        checks=json.loads((OUTPUT/'gpu_checks.json').read_text())
        if not checks.get('passed'):raise RuntimeError('Numerical preflight has not passed')
    if args.stage in ['train','all']:
        for name in selected:
            if name=='e2':continue
            if args.resume and (OUTPUT/name/'training_complete.json').exists():continue
            jobs[name]=launch('train',mapping[name],experiment=name,
                               resume=args.resume and (OUTPUT/name/'checkpoint_last.pt').exists())
    if args.stage in ['calibrate','all'] or (args.stage=='train' and 'e2' in selected):
        if not (OUTPUT/'calibration.json').exists():wait(launch('calibrate',[mapping['e2'][0]]))
    if args.stage in ['train','all'] and 'e2' in selected:
        if not (args.resume and (OUTPUT/'e2/training_complete.json').exists()):
            jobs['e2']=launch('train',mapping['e2'],experiment='e2',
                               resume=args.resume and (OUTPUT/'e2/checkpoint_last.pt').exists())
    for name,proc in jobs.items():
        wait(proc);print(json.dumps({'training_finished':name}),flush=True)
    if args.stage in ['evaluate','all']:
        eval_jobs=[]
        for name in ['baseline',*selected]:
            gpu=mapping['e1'] if name=='baseline' else mapping[name]
            # Sequential within reused cards, parallel across experiment groups.
            if name=='baseline':wait(launch('evaluate',[gpu[0]],experiment=name,resume=args.resume))
            else:eval_jobs.append(launch('evaluate',[gpu[0]],experiment=name,resume=args.resume))
        for proc in eval_jobs:wait(proc)
    if args.stage in ['report','all']:
        from tools.three_way_report import report
        report()


if __name__=='__main__':main()
