"""Best and final inference, common scoring and plain Chinese comparison."""
from __future__ import annotations

import copy
import csv
import json
import os
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from tools.three_way_experiment import (ROOT,OUTPUT,BASELINE,DATA,PRIORITY,build_model,
    load_operator,experiment_forward,experiment_loss,write_json,sha256,install_data)
from tools.three_way_checks import input_item
from tools.three_way_physics import CorrelatedVariance
from datasets.matlab_multivolume_dataset import DatasetItemKey,_read_targets
from losses.self_supervised_losses import TaylorH2VarianceModel,compute_self_supervised_loss
from utils.reconstruction_metrics import reconstruction_metrics
import tools.priority_validation_analysis as pa
from tools.priority_validation_common import scenes,load_config


def csv_write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    columns=list(dict.fromkeys(k for row in rows for k in row))
    tmp=path.with_suffix('.csv.tmp')
    with tmp.open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    tmp.replace(path)


def csv_read(path):
    with path.open() as f:return list(csv.DictReader(f))


def brightness_inputs(selected_sample=None):
    marker=OUTPUT/'brightness_inputs/complete.json'
    if marker.exists():return
    prefs=OUTPUT/'matlab_prefs/brightness';prefs.mkdir(parents=True,exist_ok=True)
    environment=os.environ.copy();environment['MATLAB_PREFDIR']=str(prefs)
    script=ROOT/'matlab_code/priority_validation'
    sample_argument='' if selected_sample is None else ','+str(selected_sample+1)
    expression=f"addpath('{script}');three_way_brightness_inputs('{ROOT}','{OUTPUT}'{sample_argument});"
    subprocess.run([str(ROOT.parent/'MATLAB/R2023b/bin/matlab'),'-batch',expression],
                   cwd=ROOT,env=environment,check=True)
    files=sorted((OUTPUT/'brightness_inputs').glob('*/*.mat'))
    if selected_sample is not None and len(files)!=12:
        sample=['points_z060_r01','lines_z060_r01','P09'][selected_sample]
        if len(list((OUTPUT/'brightness_inputs'/sample).glob('*.mat')))!=4:raise ValueError('Incomplete brightness sample')
        return
    if len(files)!=12:raise ValueError('Incomplete brightness RL3 generation')
    write_json(marker,{'complete':True,'raw_frame_gains':[.1,.5,1.,2.],
                      'matlab_rl_iterations':3,'files':{str(p.relative_to(OUTPUT)):sha256(p) for p in files},
                      'source_sha256':sha256(script/'three_way_brightness_inputs.m')})


def all_cases():
    result=[]
    for split,names in [('validation',['P09','V01','V02']),('test',['P07','T01','T02']),
                        ('diagnostic_train',['P08','P10'])]:
        for sample in names:
            for subset in range(1,11):
                result.append({'id':f'{sample}_subset_{subset:02d}','path':DATA/sample,
                               'subset':subset,'split':split,'sample':sample})
    _,config=load_config()
    for scene in scenes(config):
        for repeat in range(1,4):
            name=f'{scene.scene_id}_r{repeat:02d}'
            result.append({'id':name,'path':PRIORITY/'generated'/name,'subset':1,'split':'priority',
                           'sample':name,'family':scene.family,'scene_id':scene.scene_id,'repeat':repeat,
                           'truth_depth_um':scene.depth_um})
    result.append({'id':'zero_control','path':PRIORITY/'generated/zero_control','subset':1,
                   'split':'zero','sample':'zero_control'})
    return result


def targets(case,device):
    key=DatasetItemKey(case['sample'],case['subset'],case['split'],case['path'])
    raw=_read_targets(key,include_ground_truth=case['split']!='zero')
    return {k:torch.from_numpy(v).float().unsqueeze(0).to(device) if k in
            ['measured_mean','measured_variance','ground_truth'] else v for k,v in raw.items()}


def brightness_item(sample,gain,base,device):
    if gain==0:
        item=dict(base)
        for key in ['f_var','f_var_feature','g_mean','input_mean','residual_frames']:item[key]=base[key]*0
        return item
    path=OUTPUT/'brightness_inputs'/sample/f'gain_{gain:g}.mat'
    with h5py.File(path,'r') as f:
        fvar=pa._mat_yxz(f,'f_var');gmean=pa._mat_yxz(f,'g_mean');mu=pa._mat_yx(f,'input_mean')
        frames=np.asarray(f['input_frames']).transpose(0,2,1)
    item=dict(base)
    for key,value in [('f_var',fvar[None]),('f_var_feature',fvar[None]),('g_mean',gmean[None]),
                      ('input_mean',mu[None]),('residual_frames',(frames-mu)[:,None,:,:])]:
        item[key]=torch.from_numpy(np.ascontiguousarray(value)).float().unsqueeze(0).to(device)
    return item


def evaluate(experiment,resume=False):
    torch.set_num_threads(4);torch.cuda.set_device(0);install_data();device=torch.device('cuda:0')
    if experiment=='baseline':brightness_inputs()
    if not (OUTPUT/'brightness_inputs/complete.json').exists():raise ValueError('Generate brightness inputs first')
    folder=BASELINE if experiment=='baseline' else OUTPUT/experiment
    result_dir=OUTPUT/'evaluation'/experiment;result_dir.mkdir(parents=True,exist_ok=True)
    _,geometry=load_config();cases=all_cases()
    cfg2=yaml.safe_load((OUTPUT/'e2.yaml').read_text())
    operator=load_operator(cfg2,device);old_variance=TaylorH2VarianceModel(operator)
    audit_options=copy.deepcopy(cfg2)
    audit_options['three_way']['covariance']['train_bank']=cfg2['three_way']['covariance']['audit_bank']
    audit_options['three_way']['covariance']['train_count']=1024
    audit=CorrelatedVariance(operator,audit_options)
    mismatch_options=copy.deepcopy(audit_options)
    mismatch_options['three_way']['covariance']['train_bank']=cfg2['three_way']['covariance']['mismatch_bank']
    mismatch=CorrelatedVariance(operator,mismatch_options)
    # This diagnostic changes correlation shape at fixed illumination variance.
    # The scale is calibrated once from the independent bank, never from a candidate.
    mismatch_bank=mismatch.active_bank()
    mismatch.sigma=float(mismatch_bank.var(dim=0,unbiased=True).mean().sqrt())
    if experiment=='baseline':
        from tools.three_way_loss_audit import run as run_loss_audit
        run_loss_audit(operator,audit,old_variance)
    rows=[];brightness_rows=[];points=[];pairs=[];lines=[];profiles=[];axial=[];audit_rows=[]
    for role,filename in [('best','checkpoint_best.pt'),('final','checkpoint_last.pt')]:
        checkpoint_path=folder/filename;checkpoint_hash=sha256(checkpoint_path)
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        config=checkpoint['config'];step=int(checkpoint['completed_steps']);method=f'{experiment}_{role}'
        if role=='final' and step!=200:raise ValueError('Final checkpoint weights are not step 200')
        model=build_model(config).to(device);model.load_state_dict(checkpoint['model_state']);model.eval()
        vm=CorrelatedVariance(operator,config) if experiment=='e2' else old_variance
        if experiment=='e2':vm.evaluating=True
        for case in cases:
            destination=result_dir/role/case['id'];destination.mkdir(parents=True,exist_ok=True)
            marker=destination/'complete.json'
            # Inference consumes only ten-frame inputs. GT and holdout are loaded afterwards.
            if resume and marker.exists():
                saved_record=json.loads(marker.read_text())
                if saved_record['checkpoint_sha256']!=checkpoint_hash:raise ValueError('Checkpoint changed on resume')
                if sha256(destination/'reconstruction.npy')!=saved_record['prediction_sha256']:raise ValueError('Prediction artifact changed')
                prediction=np.load(destination/'reconstruction.npy')
                rows.append(saved_record['metrics'])
                audit_rows.extend(saved_record.get('covariance_audits',[]))
                if case['split']=='priority':truth=pa._read_yxz(case['path']/'prepared.mat','ground_truth')
            else:
                item=input_item(case['path'],case['subset'],device)
                with torch.inference_mode():
                    out,beta0=experiment_forward(model,item,operator,config=config)
                    prediction=out.reconstruction[0,0].cpu().numpy()
                    if resume and marker.exists():
                        old=json.loads(marker.read_text())
                        if old['checkpoint_sha256']!=checkpoint_hash:raise ValueError('Checkpoint changed on resume')
                        if not np.allclose(np.load(destination/'reconstruction.npy'),prediction,rtol=1e-5,atol=1e-9):
                            raise ValueError('Prediction changed on resume')
                    np.save(destination/'reconstruction.npy',prediction)
                    np.save(destination/'anchor.npy',(item['f_var']*out.beta[:,None,None,None,None])[0,0].cpu().numpy())
                    target=targets(case,device);scoring={**item,**target}
                    common=compute_self_supervised_loss(out.reconstruction,target['measured_mean'],
                             target['measured_variance'],operator,old_variance,**config['loss'])
                    native=experiment_loss(out,scoring,operator,vm,config)
                    case_audits=[]
                    row={'method':method,'experiment':experiment,'checkpoint_role':role,'weight_step':step,
                         'sample_id':case['sample'],'case_id':case['id'],'subset':case['subset'],'split':case['split'],
                         'common_mean_loss':float(common.normalized_mean),'common_variance_loss':float(common.normalized_var),
                         'native_total_loss':float(native.total),'native_mean_loss':float(native.normalized_mean),
                         'native_variance_loss':float(native.normalized_var),'native_weighted_tv':float(native.weighted_tv),
                         'output_max':float(prediction.max()),
                         'output_sum':float(prediction.sum()),'beta0':float(beta0),'beta':float(out.beta)}
                    if case['split']!='zero':
                        truth=target['ground_truth'][0,0].cpu().numpy()
                        row.update(pa._score_structure_metrics(prediction,truth))
                        from scipy.ndimage import binary_dilation
                        xy_support=binary_dilation(truth.max(0)>.1*max(float(truth.max()),1e-30),iterations=2)
                        row['background_xy_mass_fraction']=float(prediction[:,~xy_support].sum()/max(float(prediction.sum()),1e-30))
                    # Independent audit and mismatched NA scores on the three fixed diagnostics.
                    if case['sample'] in ['P08','P09','P10'] and case['subset']==1:
                        for name,modelvar in [('audit_covariance_loss',audit),('mismatch_NA04479_covariance_loss',mismatch)]:
                            predicted=modelvar(out.reconstruction,target['measured_mean'])
                            row[name]=float(torch.nn.functional.smooth_l1_loss(torch.log(predicted.clamp_min(0)+1e-6),
                                             torch.log(target['measured_variance']+1e-6)))
                            for repeat in [1,2,3]:
                                mu90,var90=pa._target_statistics(PRIORITY,case['sample'],repeat)
                                var90=torch.from_numpy(var90).to(device)[None,None]
                                score=float(torch.nn.functional.smooth_l1_loss(torch.log(predicted.clamp_min(0)+1e-6),torch.log(var90+1e-6)))
                                case_audits.append({'method':method,'sample_id':case['sample'],'repeat':repeat,
                                    'score_model':name,'variance_loss':score,'bank_sigma':modelvar.sigma,
                                    'target':'new independent 90-frame acquisition','candidate_gain_refit':False})
                        audit_rows.extend(case_audits)
                    if case['id']=='P09_subset_01':
                        perturbed=dict(item)
                        perturbed['ground_truth']=torch.ones_like(out.reconstruction)*999
                        perturbed['measured_mean']=torch.ones_like(item['input_mean'])*999
                        perturbed['measured_variance']=torch.ones_like(item['input_mean'])*999
                        repeated,_=experiment_forward(model,item,operator,config=config)
                        checked,_=experiment_forward(model,perturbed,operator,config=config)
                        norm=out.reconstruction.norm().clamp_min(1e-30)
                        repeat_relative=float((repeated.reconstruction-out.reconstruction).norm()/norm)
                        leakage_relative=float((checked.reconstruction-out.reconstruction).norm()/norm)
                        leakage=float((checked.reconstruction-out.reconstruction).abs().max())
                        # Sparse GPU summation changes analytic beta0 by one FP32 ULP on
                        # repeated identical inputs. Measure that control, not bit equality.
                        if max(repeat_relative,leakage_relative)>1e-6:
                            raise ValueError('GT/holdout contamination or repeatability check failed')
                        row['same_input_repeat_relative_l2']=repeat_relative
                        row['gt_target_contamination_relative_l2']=leakage_relative
                        row['gt_target_contamination_max_change']=leakage
                    record={'complete':True,'checkpoint_sha256':checkpoint_hash,'weight_step':step,
                            'checkpoint_role':role,'inference_target_or_gt_used':False,
                            'input_indices':np.asarray(item['input_indices']).tolist(),
                            'prediction_sha256':sha256(destination/'reconstruction.npy'),'metrics':row,'covariance_audits':case_audits}
                    write_json(marker,record);rows.append(row)
            if case['split']=='priority':
                methods={method:prediction}
                if case['family']=='points':
                    p,pp,pr=pa._point_metrics(geometry,case['scene_id'],case['repeat'],truth,methods)
                    points.extend(p);pairs.extend(pp);profiles.extend(pr)
                elif case['family']=='lines':lines.extend(pa._line_metrics(geometry,case['scene_id'],case['repeat'],methods))
                else:
                    p,pp=pa._axial_metrics(geometry,case['repeat'],methods);axial.extend(p);pairs.extend(pp)
            print(json.dumps({'evaluated':method,'case':case['id'],'weight_step':step}),flush=True)
        for sample,directory in [('points_z060_r01',PRIORITY/'generated/points_z060_r01'),
                                 ('lines_z060_r01',PRIORITY/'generated/lines_z060_r01'),('P09',DATA/'P09')]:
            base=input_item(directory,1,device)
            case={'sample':sample,'subset':1,'split':'brightness','path':directory}
            target=targets(case,device);reference=None;stored={}
            for gain in [1.,0.,.1,.5,2.]:
                item=brightness_item(sample,gain,base,device)
                with torch.inference_mode():
                    out,_=experiment_forward(model,item,operator,config=config)
                    pred=out.reconstruction
                    if gain==1.:reference=pred.clone()
                    cm=compute_self_supervised_loss(pred,target['measured_mean']*gain,target['measured_variance']*gain**2,
                                                  operator,old_variance,**config['loss'])
                    equivariance=float((pred-reference*gain).norm()/(reference*gain).norm().clamp_min(1e-30)) if gain else None
                    brightness_rows.append({'method':method,'sample_id':sample,'gain':gain,
                         'output_max':float(pred.max()),'output_sum':float(pred.sum()),
                         'relative_scale_error':equivariance,'mean_loss':float(cm.normalized_mean),
                         'variance_loss':float(cm.normalized_var),'raw_frames_scaled_and_RL3_rerun':gain!=0})
                    bdir=result_dir/role/'brightness'/sample;bdir.mkdir(parents=True,exist_ok=True)
                    np.save(bdir/f'gain_{gain:g}.npy',pred[0,0].cpu().numpy())
        del model,vm
        torch.cuda.empty_cache()
    for name,values in [('metrics',rows),('brightness',brightness_rows),('point_targets',points),('point_pairs',pairs),
                        ('lines',lines),('depth_profiles',profiles),('axial_points',axial),('covariance_audit',audit_rows)]:
        csv_write(result_dir/(name+'.csv'),values)
    write_json(result_dir/'complete.json',{'complete':True,'experiment':experiment,'cases_per_checkpoint':len(cases),
                 'checkpoints':['best','final'],'actual_weight_steps_used':True,'brightness_cases':len(brightness_rows)})


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    analysis=OUTPUT/'analysis';analysis.mkdir(parents=True,exist_ok=True)
    tables={}
    for name in ['metrics','brightness','point_targets','point_pairs','lines','depth_profiles','axial_points','covariance_audit']:
        rows=[]
        for experiment in ['baseline','e1','e2','e3']:
            folder=OUTPUT/'evaluation'/experiment
            if not (folder/'complete.json').exists():raise ValueError(f'Missing evaluation: {experiment}')
            rows.extend(csv_read(folder/(name+'.csv')))
        tables[name]=rows;csv_write(analysis/(name+'.csv'),rows)
    methods=[f'{e}_{r}' for e in ['baseline','e1','e2','e3'] for r in ['best','final']]
    primary=.1;summary=[]
    for method in methods:
        selected=[r for r in tables['metrics'] if r['method']==method]
        test=[r for r in selected if r['split']=='test']
        validation=[r for r in selected if r['split']=='validation']
        pp=[r for r in tables['point_targets'] if r['method']==method and float(r['threshold'])==primary]
        single=[r for r in pp if r['kind']=='single']
        pairs=[r for r in tables['point_pairs'] if r['method']==method and float(r['threshold'])==primary]
        line=[r for r in tables['lines'] if r['method']==method and float(r['threshold'])==primary]
        zero=next(r for r in selected if r['split']=='zero')
        avg=lambda rows,k:float(np.mean([float(r[k]) for r in rows])) if rows else float('nan')
        rate=lambda rows,k:float(np.mean([r[k] in ['True','true','1'] for r in rows])) if rows else float('nan')
        row={'method':method,'weight_step':int(selected[0]['weight_step']),
             'test_aligned_nrmse':avg(test,'gt_scale_aligned_nrmse'),'test_axial_w1_um':avg(test,'gt_axial_w1_um'),
             'test_local_axial_w1_um':avg(test,'local_axial_w1_um'),'test_background_xy_mass_fraction':avg(test,'background_xy_mass_fraction'),
             'test_mean_loss':avg(test,'common_mean_loss'),'test_variance_loss':avg(test,'common_variance_loss'),
             'validation_aligned_nrmse':avg(validation,'gt_scale_aligned_nrmse'),
             'single_localization_rate':rate(single,'matched'),'local_depth_w1_um':avg(pp,'local_depth_w1_um'),
             'lateral_pair_separation_rate':rate([r for r in pairs if r['scene_id'].startswith('points')],'separated'),
             'axial_pair_separation_rate':rate([r for r in pairs if r['scene_id']=='axial_pairs'],'separated'),
             'continuous_false_gap_rate':rate([r for r in line if float(r['gap_um'])==0],'false_gap'),
             'broken_bridge_rate':rate([r for r in line if float(r['gap_um'])>0],'bridged'),
             'zero_max':float(zero['output_max']),'zero_sum':float(zero['output_sum'])}
        br=[r for r in tables['brightness'] if r['method']==method and float(r['gain']) in [.1,.5,2.]]
        row['brightness_relative_scale_error']=avg(br,'relative_scale_error');summary.append(row)
    csv_write(analysis/'summary.csv',summary);write_json(analysis/'summary.json',summary)
    # Every repeat is drawn separately; no averaging of predictions or per-layer rescaling.
    fig,axes=plt.subplots(2,3,figsize=(17,9))
    for ax,key,title in zip(axes.ravel(),['test_aligned_nrmse','test_axial_w1_um','brightness_relative_scale_error','continuous_false_gap_rate','broken_bridge_rate','test_background_xy_mass_fraction'],
                            ['Test shape error (lower better)','Test axial W1, um (lower better)','Brightness scaling error',
                             'False breaks in continuous lines','False connections across gaps','Background mass fraction']):
        ax.bar(methods,[r[key] for r in summary]);ax.set_title(title);ax.tick_params(axis='x',rotation=65)
    fig.tight_layout();fig.savefig(analysis/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(16,8),sharex=True,sharey=True)
    for ax,method in zip(axes.ravel(),methods):
        for repeat in [1,2,3]:
            for cell in sorted({r['cell_id'] for r in tables['point_targets'] if r['kind']=='single'}):
                rows=[r for r in tables['point_targets'] if r['method']==method and r['kind']=='single' and
                      int(r['repeat'])==repeat and r['cell_id']==cell and float(r['threshold'])==primary]
                rows.sort(key=lambda r:float(r['z_um']))
                if rows:ax.plot([float(r['z_um']) for r in rows],
                                 [float(r.get('pred_z_layer','nan') or 'nan')*10+10 for r in rows],alpha=.55)
        ax.plot([10,100],[10,100],'k--');ax.set_title(method);ax.set_xlabel('True depth, um');ax.set_ylabel('Predicted depth, um')
    fig.tight_layout();fig.savefig(analysis/'depth_following_all_repeats.png',dpi=160);plt.close(fig)
    from tools.three_way_figures import make as make_additional_figures
    figure_record=make_additional_figures(tables,summary)
    write_json(analysis/'figure_manifest.json',figure_record)
    lines=['# 三组实验验证结果','',
      '本轮沿用原数据：照明 NA=0.05、成像 PSF NA=0.15。是否对应真实系统尚未确认。',
      '三组均从相同随机初始权重训练 200 步，全局 batch=8。best 按各自验证目标选取；final 是第 200 步的实际权重。',
      'E1：十帧整体亮度归一化后重建，再乘回尺度。E2：方差前向加入同层与跨层散斑相关。E3：均值只调亮度，方差只调结构。','',
      '| 方法 | 权重步数 | 测试集结构误差↓ | 测试集深度误差 µm↓ | 连续线假断口↓ | 黑输入最大值↓ |',
      '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['method']} | {r['weight_step']} | {r['test_aligned_nrmse']:.4f} | {r['test_axial_w1_um']:.2f} | {r['continuous_false_gap_rate']:.1%} | {r['zero_max']:.5g} |")
    lines+=['','结构误差采用统一的尺度对齐指标，只用于评价形状；对齐增益不用于修改正式预测。亮度是否正确另看均值误差及亮度缩放实验。',
            '各组 native loss 定义不同，不能按其绝对数值跨组排名。共同的均值、旧方差评分和独立相关模型评分保存在逐项 CSV 中。',
            '测试对象未参与本轮梯度更新或 checkpoint 选择；这些对象已用于此前分析，因此不能称为从未看过的新测试集。',
            '点、点对、线条和断口保留全部三次独立采集；原对象十个重叠子集不作为独立重复。','',
            '![统一比较](analysis/comparison.png)','![逐次深度跟随](analysis/depth_following_all_repeats.png)','',
            '原始预测：evaluation/<实验>/<best或final>/<场景>/reconstruction.npy。逐位置剖面、全部阈值、漏检及假峰结果：analysis/*.csv。',
            '来源哈希及训练条件：preflight.json、source_snapshot/、e1.yaml/e2.yaml/e3.yaml。',
            '恢复运行：`python tools/run_three_way_validation.py --stage all --resume`。']
    (OUTPUT/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    write_json(OUTPUT/'complete.json',{'complete':True,'experiments':['e1','e2','e3'],'baseline_reused':True,
                'best_and_final_evaluated':True,'summary':summary,'finished_unix':time.time()})

    from tools.three_way_conclusions import write_report as write_interpreted_report
    write_interpreted_report()
