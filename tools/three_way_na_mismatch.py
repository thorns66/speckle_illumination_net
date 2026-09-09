"""Paired acquisition with a changed illumination pupil; frozen-model supplement."""
import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
import torch
import yaml
from scipy.io import savemat
from tools.three_way_experiment import ROOT,OUTPUT,PRIORITY,BASELINE,load_operator,write_json,sha256,build_model,experiment_forward,install_data
from tools.three_way_checks import input_item
from tools.three_way_report import csv_write,csv_read,targets
from tools.priority_validation_common import load_config
import tools.priority_validation_analysis as pa

SAMPLES=['points_z060_r01','lines_z060_r01','axial_pairs_r01']

def generate():
    torch.set_num_threads(4);torch.cuda.set_device(0)
    config=yaml.safe_load((OUTPUT/'e2.yaml').read_text())
    operator=load_operator(config,torch.device('cuda:0'))
    bank_path=OUTPUT/'illumination/mismatch_acquisition_na04479.mat'
    with h5py.File(bank_path,'r') as f:
        bank=np.asarray(f['illumination_raw'],dtype=np.float32).transpose(0,1,3,2).copy()
        phase_error=float(np.asarray(f['illumination_meta/paired_phase_reference_relative_l2']).item())
    assert phase_error<1e-6
    for sample in SAMPLES:
        folder=OUTPUT/'mismatch_generated'/sample.replace('_r01','_na04479_r01')
        source=PRIORITY/'generated'/sample
        marker=folder/'frames_complete.json'
        if marker.exists():
            saved=json.loads(marker.read_text())
            for name,digest in saved['files'].items():
                if sha256(folder/name)!=digest:raise ValueError('Changed mismatch data on resume')
            continue
        truth=pa._read_yxz(source/'prepared.mat','ground_truth')
        density=torch.from_numpy(truth).to('cuda:0')[None,None]
        sensor=[]
        with torch.inference_mode():
            for start in range(0,100,16):
                illum=torch.from_numpy(bank[start:start+16]).to('cuda:0')[:,None]
                sensor.append(operator(illum*density)[:,0].cpu().numpy())
        frames=np.concatenate(sensor)
        folder.mkdir(parents=True,exist_ok=True);(folder/'sensor_frames').mkdir(exist_ok=True)
        for i,frame in enumerate(frames,1):
            # These raw-frame files are consumed by Python only. The RL3 input below is a MATLAB MAT file.
            with h5py.File(folder/'sensor_frames'/f'frame_{i:03d}.mat','w') as f:
                f.create_dataset('sensor_pre_detector',data=frame.T)
        shutil.copy2(source/'prepared.mat',folder/'prepared.mat')
        statistics={'input_indices':np.arange(1,11),'holdout_indices':np.arange(11,101),'z_um':np.arange(10,101,10)}
        for prefix,selected in [('input',frames[:10]),('holdout',frames[10:])]:
            statistics[prefix+'_physics_mean_float']=selected.mean(0,dtype=np.float64).astype(np.float32)
            statistics[prefix+'_physics_variance_nminus1_float']=selected.var(0,ddof=1,dtype=np.float64).astype(np.float32)
        savemat(folder/'input_statistics.mat',statistics)
        write_json(marker,{'complete':True,'paired_NA05_case':sample,'NA':.04479,'detection_NA':.15,
             'illumination_sha256':sha256(bank_path),'source_GT_sha256':sha256(source/'prepared.mat'),
             'forward':'verified exact sparse LFM; no camera noise','independent_repeats':1,
             'frames_format':'raw frame HDF5 for Python; statistics native MAT v5; RL3 native MAT v7.3',
             'files':{str(p.relative_to(folder)):sha256(p) for p in folder.rglob('*.mat')}})
        print('Mismatch frames ready: '+sample,flush=True)
    del operator;torch.cuda.empty_cache()
    prefs=OUTPUT/'matlab_prefs/mismatch_reconstruct';prefs.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['MATLAB_PREFDIR']=str(prefs)
    for sample in SAMPLES:
        name=sample.replace('_r01','_na04479_r01')
        expr=f"addpath('{ROOT}/matlab_code/priority_validation');three_way_mismatch_reconstruct('{ROOT}','{OUTPUT}','{name}');"
        subprocess.run([str(ROOT.parent/'MATLAB/R2023b/bin/matlab'),'-batch',expr],env=env,check=True,cwd=ROOT)
    write_json(OUTPUT/'mismatch_generated/complete.json',{'complete':True,'samples':SAMPLES,'independent_repeats':1,
         'paired_phase_reference_relative_l2':phase_error,'models_retrained':False})

def evaluate():
    torch.set_num_threads(4);torch.cuda.set_device(0);install_data()
    config=yaml.safe_load((OUTPUT/'e2.yaml').read_text());operator=load_operator(config,torch.device('cuda:0'))
    _,geometry=load_config();rows=[];points=[];pairs=[];lines=[];profiles=[];axial=[]
    for experiment in ['baseline','e1','e2','e3']:
        folder=BASELINE if experiment=='baseline' else OUTPUT/experiment
        for role,filename in [('best','checkpoint_best.pt'),('final','checkpoint_last.pt')]:
            ckpt=torch.load(folder/filename,map_location='cpu',weights_only=False)
            model=build_model(ckpt['config']).to('cuda:0');model.load_state_dict(ckpt['model_state']);model.eval()
            method=experiment+'_'+role
            for sample in SAMPLES:
                path=OUTPUT/'mismatch_generated'/sample.replace('_r01','_na04479_r01')
                item=input_item(path,1,'cuda:0')
                with torch.inference_mode():out,_=experiment_forward(model,item,operator,config=ckpt['config'])
                pred=out.reconstruction[0,0].detach().cpu().numpy()
                dest=OUTPUT/'mismatch_evaluation'/method/sample;dest.mkdir(parents=True,exist_ok=True)
                np.save(dest/'reconstruction.npy',pred)
                truth=pa._read_yxz(path/'prepared.mat','ground_truth')
                row={'method':method,'paired_case_id':sample,'weight_step':ckpt['completed_steps'],
                     'illumination_NA':.04479,'repeat':1,**pa._score_structure_metrics(pred,truth)}
                matched_path=OUTPUT/'evaluation'/experiment/role/sample/'reconstruction.npy'
                if matched_path.exists():
                    matched=pa._score_structure_metrics(np.load(matched_path),truth)
                    for key in ['gt_scale_aligned_nrmse','gt_axial_w1_um','local_axial_w1_um']:
                        row['matched_NA05_'+key]=matched[key];row['change_'+key]=row[key]-matched[key]
                rows.append(row)
                if sample.startswith('points'):
                    p,pp,pr=pa._point_metrics(geometry,'points_z060',1,truth,{method:pred});points.extend(p);pairs.extend(pp);profiles.extend(pr)
                elif sample.startswith('lines'):lines.extend(pa._line_metrics(geometry,'lines_z060',1,{method:pred}))
                else:
                    p,pp=pa._axial_metrics(geometry,1,{method:pred});axial.extend(p);pairs.extend(pp)
                write_json(dest/'complete.json',{'checkpoint_sha256':sha256(folder/filename),'prediction_sha256':sha256(dest/'reconstruction.npy'),
                     'input_only_inference':True,'input_indices':list(range(1,11)),'weight_step':ckpt['completed_steps']})
                print('Mismatch evaluated: '+method+' '+sample,flush=True)
            del model;torch.cuda.empty_cache()
    for name,data in [('metrics',rows),('points',points),('pairs',pairs),('lines',lines),('profiles',profiles),('axial',axial)]:
        csv_write(OUTPUT/'analysis'/('na_mismatch_'+name+'.csv'),data)
    write_json(OUTPUT/'mismatch_evaluation/complete.json',{'complete':True,'prediction_count':len(rows),
          'one_paired_repeat_only':True,'illumination_NA':.04479,'matched_NA':.05,'trained_models_unchanged':True})

if __name__=='__main__':
    generate();evaluate()
