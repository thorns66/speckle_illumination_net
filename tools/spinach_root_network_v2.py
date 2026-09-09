"""Frozen E3+mean<=100% inference; full field first, fixed tile fallback on OOM."""
from __future__ import annotations
import argparse,json,math,os
from pathlib import Path
import h5py,numpy as np,torch,tifffile
from scipy.io import savemat
from tools import mean_budget_experiment as experiment
from tools.three_way_checks import input_item

ROOT=Path(__file__).resolve().parents[1]

def mat_volume(path,name):
    with h5py.File(path,'r') as h:return np.asarray(h[name],dtype=np.float32).transpose(0,2,1).copy()

def regression(model,config,device,cfg):
    operator=experiment.load_operator(config,device)
    sample=ROOT/'data/speckle_dataset_v3_full_20260907_run01/T03'
    item=input_item(sample,1,device)
    with torch.inference_mode():out,_=experiment.forward(model,item,operator,config=config)
    observed=out.reconstruction[0,0].cpu().numpy()
    reference=np.load(ROOT/'outputs/v3_mean050_mean100_400_20260908_run01/evaluation/e3_mean100/final/T03_subset_01/reconstruction.npy')
    error=float(np.linalg.norm(observed-reference)/max(np.linalg.norm(reference),1e-30))
    if error>1e-5:raise RuntimeError(f'Frozen inference regression failed: {error}')
    Path(cfg['network_regression']).write_text(json.dumps({'complete':True,'case':'T03_subset_01',
        'relative_l2':error},indent=2));del operator,item,out,observed;torch.cuda.empty_cache()

def execute(model,fvar,gmean,residual,z,beta0,device):
    def t(a):return torch.from_numpy(a).to(device=device,dtype=torch.float32)
    try:
        torch.cuda.reset_peak_memory_stats(device)
        fv=t(fvar)[None,None];gm=t(gmean)[None,None];rr=t(residual)[None,:,None];zz=t(z)[None];b=t(np.array([beta0],np.float32))
        with torch.inference_mode():out=model(fv,gm,rr,zz,beta0=b,var_feature_volume=fv)
        value=out.reconstruction[0,0].cpu().numpy()
        return value,{'mode':'full_field','fallback':False,'peak_memory_bytes':torch.cuda.max_memory_allocated(device)}
    except torch.OutOfMemoryError:
        for name in ('fv','gm','rr','zz','b','out'):
            if name in locals():del locals()[name]
        torch.cuda.empty_cache()
    height,width=fvar.shape[-2:];core=392;context=98;value=np.empty_like(fvar);peak=0
    with torch.inference_mode():
        for y0 in range(0,height,core):
            for x0 in range(0,width,core):
                y1=min(y0+core,height);x1=min(x0+core,width);ay0=max(0,y0-context);ax0=max(0,x0-context);ay1=min(height,y1+context);ax1=min(width,x1+context)
                fv=t(fvar[:,ay0:ay1,ax0:ax1])[None,None];gm=t(gmean[:,ay0:ay1,ax0:ax1])[None,None]
                rr=t(residual[:,ay0:ay1,ax0:ax1])[None,:,None];zz=t(z)[None];b=t(np.array([beta0],np.float32))
                out=model(fv,gm,rr,zz,beta0=b,var_feature_volume=fv).reconstruction[0,0]
                value[:,y0:y1,x0:x1]=out[:,y0-ay0:y1-ay0,x0-ax0:x1-ax0].cpu().numpy()
                peak=max(peak,torch.cuda.max_memory_allocated(device))
    return value,{'mode':'context_tiled','fallback':True,'core':core,'context':context,
      'phase_period':49,'peak_memory_bytes':peak,'limitation':'GroupNorm statistics are tile-local'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);a=p.parse_args();cfg=json.loads(a.manifest.read_text())
    assert os.environ['CUDA_VISIBLE_DEVICES']==cfg['gpu_uuid']['network'];device=torch.device('cuda:0');torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    checkpoint=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False);config=checkpoint['config']
    assert cfg['checkpoint_role']=='final' and checkpoint['completed_steps']==cfg['checkpoint_step']==400
    assert config['v3_compare']['kind']=='e3_mean100' and float(config['v3_compare']['shape_gradient_budget'])==1
    model=experiment.build_model(config,initial=False).to(device);model.load_state_dict(checkpoint['model_state'],strict=True);model.eval()
    regression(model,config,device,cfg)
    mean=mat_volume(cfg['output_mat']['mean'],'reconstruction_raw');taylor=mat_volume(cfg['output_mat']['taylor'],'reconstruction_sqrt')
    with h5py.File(cfg['output_mat']['taylor'],'r') as h:projection=np.asarray(h['projection_of_sqrt'],dtype=np.float32).T.copy()
    frames=np.stack([tifffile.imread(path).astype(np.float32)/255 for path in cfg['selected_files']])
    image_mean=frames.mean(0,dtype=np.float64).astype(np.float32);residual=frames-image_mean
    beta0=float(np.sum(projection.astype(np.float64)*image_mean)/max(np.sum(projection.astype(np.float64)**2)+1e-8,1e-30))
    if not np.isfinite(beta0) or beta0<=0:raise ValueError('Invalid global beta0')
    base,mode=execute(model,taylor,mean,residual,np.asarray(cfg['z_um'],np.float32),beta0,device)
    if not np.isfinite(base).all() or np.any(base<0):raise ValueError('Invalid network output')
    savemat(cfg['network_base_mat'],{'base_reconstruction':base.transpose(1,2,0)},do_compression=False)
    factor=1+float(config['three_way']['gain_bound'])*math.tanh(float(model.mean_gain_gamma.detach().cpu()))
    cfg['e3_gain_factor']=factor;a.manifest.write_text(json.dumps(cfg,indent=2,ensure_ascii=False))
    Path(cfg['network_record']).write_text(json.dumps({'complete':True,'beta0':beta0,
      'raw_beta':float(model.raw_beta.detach().cpu()),'mean_gain_gamma':float(model.mean_gain_gamma.detach().cpu()),
      'e3_gain_factor':factor,**mode},indent=2))

if __name__=='__main__':main()
