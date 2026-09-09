"""Dated, resumable supervisor for the three-method spinach-root run."""
from __future__ import annotations
import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import tifffile
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from utils.experiment_paths import next_experiment_path
from tools.report_spinach_root import report

DATA=ROOT/'data/菠菜根'
CHECKPOINT=ROOT/'outputs/v3_mean050_mean100_400_20260908_run01/e3_mean100/checkpoint_last.pt'
PSF=ROOT/'psf/NEW_modifyfobj_PSFmatrix_M4NA0.15MLPitch220fml4000OSR3chunk05from10to130zspacing14.6154Nnum49lambda532n1a0_-11b0_2.9333.mat'
MATLAB=Path('/workspace/xyx/MATLAB/R2023b/bin/matlab')

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()

def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str))

def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'),message,flush=True)

def inventory():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu',
                                  '--format=csv,noheader,nounits'],text=True)
    return [dict(index=int(a),uuid=b.strip(),free_mib=int(c),utilization=int(d))
            for a,b,c,d in (line.split(',') for line in text.splitlines())]

def launch_matlab(output,label,uuid,expression):
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=uuid
    env['MATLAB_PREFDIR']=str(output/'matlab_preferences'/label)
    paths=[ROOT/'matlab_code'/part for part in ('real_data','pilot_dataset','Util','Solver')]
    quote=lambda path:"'"+str(path).replace("'","''")+"'"
    command='addpath('+','.join(quote(path) for path in paths)+');'+expression
    stream=(output/f'{label}.log').open('a')
    process=subprocess.Popen([str(MATLAB),'-singleCompThread','-softwareopengl','-batch',command],
                             cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    return process,stream

def prepare(output):
    files=sorted(DATA.glob('50_*.tif'))
    assert [path.name for path in files]==[f'50_{i:03d}.tif' for i in range(1,101)]
    frames=np.stack([tifffile.imread(path) for path in files])
    assert frames.shape==(100,1029,1421) and frames.dtype==np.uint8
    indices=sorted((np.random.default_rng(20260909).choice(100,10,replace=False)+1).tolist())
    holdout=sorted(set(range(1,101))-set(indices))
    selected=frames[np.asarray(indices)-1].astype(np.float32)/255
    other=frames[np.asarray(holdout)-1].astype(np.float32)/255
    input_dir=output/'input';input_dir.mkdir()
    statistics={
        'mean':selected.mean(0,dtype=np.float64).astype(np.float32),
        'variance':selected.var(0,ddof=1,dtype=np.float64).astype(np.float32),
        'holdout_mean':other.mean(0,dtype=np.float64).astype(np.float32),
        'holdout_variance':other.var(0,ddof=1,dtype=np.float64).astype(np.float32),
    }
    for name,value in statistics.items():
        tifffile.imwrite(input_dir/f'{name}.tif',value,photometric='minisblack')
    np.save(input_dir/'selected_frames_float.npy',selected)
    current=inventory()
    candidates=sorted([gpu for gpu in current if gpu['free_mib']>=30000],
                      key=lambda gpu:(gpu['utilization']>5,-gpu['free_mib'],gpu['index']))
    if len(candidates)<4:raise RuntimeError('Four GPUs with at least 30 GiB free are required')
    chosen=candidates[:4]
    methods={'mean':str(output/'mean_rl3.mat'),'taylor':str(output/'taylor_rl3.mat')}
    cfg={
        'schema_version':2,'experiment':'spinach_root_exploratory','input_dir':str(DATA),
        'image_shape_yx':[1029,1421],'input_indices':indices,'holdout_indices':holdout,
        'selected_files':[str(files[i-1]) for i in indices],
        'holdout_files':[str(files[i-1]) for i in holdout],
        'known_limitation':'upstream per-frame max normalization and uint8 quantization',
        'psf_path':str(PSF),'psf_sha256':sha(PSF),'z_um':list(range(10,101,10)),'iterations':3,
        'checkpoint':str(CHECKPOINT),'checkpoint_sha256':sha(CHECKPOINT),
        'checkpoint_role':'final','checkpoint_step':400,
        'gpu_uuid':dict(mean=chosen[0]['uuid'],taylor=chosen[1]['uuid'],network=chosen[2]['uuid'],gain=chosen[3]['uuid']),
        'gpu_assignment':dict(mean=chosen[0]['index'],taylor=chosen[1]['index'],network=chosen[2]['index'],gain=chosen[3]['index']),
        'initial_gpu_inventory':current,'mean_tiff':str(input_dir/'mean.tif'),
        'variance_tiff':str(input_dir/'variance.tif'),'holdout_mean_tiff':str(input_dir/'holdout_mean.tif'),
        'holdout_variance_tiff':str(input_dir/'holdout_variance.tif'),'output_mat':methods,
        'network_base_mat':str(output/'network_base.mat'),'network_final_mat':str(output/'e3_mean100.mat'),
        'network_record':str(output/'network_record.json'),'network_regression':str(output/'network_regression.json'),
        'input_files_sha256':{str(path):sha(path) for path in files},
    }
    source_files=[ROOT/'tools'/name for name in ('run_spinach_root_exploratory_v2.py','spinach_root_network_v2.py','report_spinach_root.py','mean_budget_experiment.py','v3_compare_experiment.py')]
    source_files += [ROOT/'matlab_code/real_data/spinach_root_rl_worker_v2.m',ROOT/'matlab_code/real_data/spinach_root_gain_worker_v2.m']
    cfg['source_sha256']={str(path):sha(path) for path in source_files}
    save(output/'manifest.json',cfg)
    return cfg

def verify(cfg):
    assert sha(cfg['checkpoint'])==cfg['checkpoint_sha256']
    assert sha(cfg['psf_path'])==cfg['psf_sha256']
    for path,expected in {**cfg['input_files_sha256'],**cfg['source_sha256']}.items():
        if sha(path)!=expected:raise RuntimeError(f'Frozen source changed: {path}')
    checkpoint=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)
    assert checkpoint['completed_steps']==400 and checkpoint['config']['v3_compare']['kind']=='e3_mean100'

def main():
    output=next_experiment_path(ROOT/'outputs','spinach_root_exploratory')
    output.mkdir(parents=True);log(f'Output {output}')
    try:
        cfg=prepare(output);verify(cfg);manifest=output/'manifest.json'
        workers=[]
        for mode in ('mean','taylor'):
            process,stream=launch_matlab(output,mode,cfg['gpu_uuid'][mode],
                f"spinach_root_rl_worker_v2('{manifest}','{mode}');")
            workers.append((mode,process,stream))
        save(output/'active_processes.json',{mode:process.pid for mode,process,_ in workers})
        log('Mean/Taylor RL3 launched in parallel')
        for mode,process,stream in workers:
            code=process.wait();stream.close()
            if code:raise RuntimeError(f'{mode} RL failed; inspect {mode}.log')
        log('Both RL3 controls complete')
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=cfg['gpu_uuid']['network']
        env['MPLCONFIGDIR']=str(output/'mpl_cache')
        with (output/'network.log').open('a') as stream:
            result=subprocess.run([sys.executable,str(ROOT/'tools/spinach_root_network_v2.py'),
                '--manifest',str(manifest)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Network inference failed; inspect network.log')
        log('Frozen network base complete')
        cfg=json.loads(manifest.read_text())
        process,stream=launch_matlab(output,'gain',cfg['gpu_uuid']['gain'],
            f"spinach_root_gain_worker_v2('{manifest}');")
        code=process.wait();stream.close()
        if code:raise RuntimeError('E3 global gain failed; inspect gain.log')
        log('Global E3 gain and projections complete')
        report(output,manifest);verify(json.loads(manifest.read_text()))
        save(output/'complete.json',{'complete':True,
            'methods':['mean_rl3','taylor_rl3_sqrt','e3_mean100'],
            'input_indices':cfg['input_indices'],'checkpoint_role':'final','checkpoint_step':400,
            'gpu_workers_exited':True,'final_gpu_inventory':inventory()})
        log('ALL COMPLETE')
    except Exception:
        save(output/'failure.json',{'traceback':traceback.format_exc()})
        raise

if __name__=='__main__':main()
