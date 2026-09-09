"""Dated, resumable real-data exploratory comparison supervisor."""
from __future__ import annotations
import hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np,tifffile,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from utils.experiment_paths import next_experiment_path
from tools.report_spinach_root import report

DATA=ROOT/'data/菠菜根';CHECKPOINT=ROOT/'outputs/v3_mean050_mean100_400_20260908_run01/e3_mean100/checkpoint_last.pt'
PSF=ROOT/'psf/NEW_modifyfobj_PSFmatrix_M4NA0.15MLPitch220fml4000OSR3chunk05from10to130zspacing14.6154Nnum49lambda532n1a0_-11b0_2.9333.mat'
MATLAB=Path('/workspace/xyx/MATLAB/R2023b/bin/matlab')

def sha(path):
 d=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):d.update(b)
 return d.hexdigest()
def save(path,v):Path(path).write_text(json.dumps(v,indent=2,ensure_ascii=False,default=str))
def log(s):print(time.strftime('%Y-%m-%d %H:%M:%S'),s,flush=True)
def inventory():
 text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 return [dict(index=int(a),uuid=b.strip(),free_mib=int(c),utilization=int(d)) for a,b,c,d in (line.split(',') for line in text.splitlines())]
def matlab(output,label,uuid,expression):
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=uuid;env['MATLAB_PREFDIR']=str(output/'matlab_preferences'/label)
 paths=[ROOT/'matlab_code'/p for p in ('real_data','pilot_dataset','Util','Solver')]
 q=lambda p:"'"+str(p).replace("'","''")+"'";expr='addpath('+','.join(q(p) for p in paths)+');'+expression
 stream=(output/f'{label}.log').open('a');p=subprocess.Popen([str(MATLAB),'-singleCompThread','-softwareopengl','-batch',expr],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
 return p,stream
def prepare(out):
 files=sorted(DATA.glob('50_*.tif'));assert [p.name for p in files]==[f'50_{i:03d}.tif' for i in range(1,101)]
 frames=np.stack([tifffile.imread(p) for p in files]);assert frames.shape==(100,1029,1421) and frames.dtype==np.uint8
 indices=sorted((np.random.default_rng(20260909).choice(100,10,replace=False)+1).tolist());hold=sorted(set(range(1,101))-set(indices))
 selected=frames[np.array(indices)-1].astype(np.float32)/255;other=frames[np.array(hold)-1].astype(np.float32)/255
 inp=out/'input';inp.mkdir();
 stats={'mean':selected.mean(0,dtype=np.float64).astype(np.float32),'variance':selected.var(0,ddof=1,dtype=np.float64).astype(np.float32),
        'holdout_mean':other.mean(0,dtype=np.float64).astype(np.float32),'holdout_variance':other.var(0,ddof=1,dtype=np.float64).astype(np.float32)}
 for n,a in stats.items():tifffile.imwrite(inp/f'{n}.tif',a,photometric='minisblack')
 available=sorted(inventory(),key=lambda x:(x['utilization']>5,-x['free_mib'],x['index']))
 if len([g for g in available if g['free_mib']>=30000])<4:raise RuntimeError('Need four GPUs with >=30 GiB free for parallel safe execution')
 chosen=[g for g in available if g['free_mib']>=30000][:4]
 output_mat={'mean':str(out/'mean_rl3.mat'),'taylor':str(out/'taylor_rl3.mat')}
 cfg={'schema_version':1,'experiment':'spinach_root_exploratory','input_dir':str(DATA),'image_shape_yx':[1029,1421],
      'input_indices':indices,'holdout_indices':hold,'selected_files':[str(files[i-1]) for i in indices],
      'holdout_files':[str(files[i-1]) for i in hold],'known_limitation':'upstream per-frame max normalization and uint8 quantization',
      'psf_path':str(PSF),'psf_sha256':sha(PSF),'z_um':list(range(10,101,10)),'iterations':3,
      'checkpoint':str(CHECKPOINT),'checkpoint_sha256':sha(CHECKPOINT),'checkpoint_role':'final','checkpoint_step':400,
      'gpu_uuid':dict(mean=chosen[0]['uuid'],taylor=chosen[1]['uuid'],network=chosen[2]['uuid'],gain=chosen[3]['uuid']),
      'gpu_assignment':dict(mean=chosen[0]['index'],taylor=chosen[1]['index'],network=chosen[2]['index'],gain=chosen[3]['index']),
      'initial_gpu_inventory':inventory(),'mean_tiff':str(inp/'mean.tif'),'variance_tiff':str(inp/'variance.tif'),
      'holdout_mean_tiff':str(inp/'holdout_mean.tif'),'holdout_variance_tiff':str(inp/'holdout_variance.tif'),
      'output_mat':output_mat,'network_base_mat':str(out/'network_base.mat'),'network_final_mat':str(out/'e3_mean100.mat'),
      'network_record':str(out/'network_record.json'),'network_regression':str(out/'network_regression.json'),
      'input_files_sha256':{str(p):sha(p) for p in files}}
    save(out/'manifest.json',cfg);np.save(inp/'selected_frames_float.npy',selected)
    return cfg
def verify(cfg):
 assert sha(cfg['checkpoint'])==cfg['checkpoint_sha256'] and sha(cfg['psf_path'])==cfg['psf_sha256']
 for p,h in cfg['input_files_sha256'].items():assert sha(p)==h
 c=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False);assert c['completed_steps']==400 and c['config']['v3_compare']['kind']=='e3_mean100'
def main():
 out=next_experiment_path(ROOT/'outputs','spinach_root_exploratory');out.mkdir(parents=True);log(f'Output {out}')
 try:
  cfg=prepare(out);verify(cfg);manifest=out/'manifest.json'
  workers=[]
  for mode in ('mean','taylor'):
   p,s=matlab(out,mode,cfg['gpu_uuid'][mode],f"spinach_root_rl_worker_v2('{manifest}', '{mode}');");workers.append((mode,p,s))
  save(out/'active_processes.json',{m:p.pid for m,p,_ in workers});log('Mean/Taylor RL3 launched in parallel')
  for mode,p,s in workers:
   code=p.wait();s.close()
   if code:raise RuntimeError(f'{mode} RL failed')
  log('Both RL3 controls complete')
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=cfg['gpu_uuid']['network'];env['MPLCONFIGDIR']=str(out/'mpl_cache')
  with (out/'network.log').open('a') as s:
   p=subprocess.run([sys.executable,str(ROOT/'tools/spinach_root_network_v2.py'),'--manifest',str(manifest)],cwd=ROOT,env=env,stdout=s,stderr=subprocess.STDOUT)
  if p.returncode:raise RuntimeError('Network inference failed');log('Frozen network base complete')
  cfg=json.loads(manifest.read_text());p,s=matlab(out,'gain',cfg['gpu_uuid']['gain'],f"spinach_root_gain_worker_v2('{manifest}');");code=p.wait();s.close()
  if code:raise RuntimeError('E3 global gain failed')
  report(out,manifest);verify(json.loads(manifest.read_text()))
  save(out/'complete.json',{'complete':True,'methods':['mean_rl3','taylor_rl3_sqrt','e3_mean100'],
    'input_indices':cfg['input_indices'],'checkpoint_role':'final','checkpoint_step':400,
    'gpu_workers_exited':True,'final_gpu_inventory':inventory()});log('ALL COMPLETE')
 except Exception:
  save(out/'failure.json',{'traceback':traceback.format_exc()});raise

if __name__=='__main__':main()
