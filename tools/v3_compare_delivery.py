"""Complete provenance, repeat plots, gradient audit and plain-language delivery."""
from __future__ import annotations
import csv,json,math,shutil,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools import v3_compare_experiment as exp
from tools import three_way_experiment as old
from tools.v3_compare_evaluation import all_cases,csv_write
from tools.v3_compare_local_audit import t03,t04,v03
from training.global_batch_schedule import FixedGlobalBatchScheduler
from datasets.matlab_multivolume_dataset import load_dataset_index,DatasetItemKey,_read_targets

OUT=exp.OUTPUT
A=OUT/'analysis'

def read(name):
    with (A/(name+'.csv')).open() as f:return list(csv.DictReader(f))

def mean(rows,key):
    vals=[]
    for r in rows:
        try:v=float(r[key])
        except (KeyError,ValueError,TypeError):continue
        if math.isfinite(v):vals.append(v)
    return float(np.mean(vals)) if vals else float('nan')

def truthy(v):return str(v).lower() in ('true','1')

def rate(rows,key):return float(np.mean([truthy(r[key]) for r in rows])) if rows else float('nan')

def gradient_audit():
    indexed,_=load_dataset_index(exp.DATA)
    sampler=FixedGlobalBatchScheduler(110,8,20260901)
    batches={step:sampler.next_batch() for step in range(1,401)}
    rows=[];summary=[]
    for arm in ('e3','e3_mean005'):
        collected=[]
        for p in sorted((OUT/arm).glob('gradient_diagnostics_rank*.jsonl')):
            collected.extend(json.loads(line) for line in p.read_text().splitlines() if line.strip())
        collected=[r for r in collected if r['phase']=='training']
        assert len(collected)==3200,(arm,len(collected))
        bystep=defaultdict(list)
        for r in collected:bystep[int(r['update_step'])].append(r)
        assert set(bystep)==set(range(1,401))
        world=json.loads((OUT/arm/'run_contract.json').read_text())['world_size']
        for step,items in sorted(bystep.items()):
            assert len(items)==8,(arm,step,len(items))
            expected=[indexed['train'][i] for i in batches[step]]
            assert sorted((r['sample_id'],r['subset_index']) for r in items)==sorted((k.sample_id,k.subset_index) for k in expected)
            for rank in range(world):
                observed=sorted([r for r in items if r['rank']==rank],key=lambda x:x['micro_call'])
                assert [(r['sample_id'],r['subset_index']) for r in observed]==[(k.sample_id,k.subset_index) for k in expected[rank::world]]
            budget=(.05 if arm=='e3_mean005' else 0)*min(step/50,1)
            assert all(r['budget_passed'] and r['gradients_finite'] and r['shape_gradient_ratio']<=budget+1e-7 for r in items)
            rows.append(dict(experiment=arm,step=step,samples=8,budget=budget,
                             mean_ratio=np.mean([r['shape_gradient_ratio'] for r in items]),
                             max_ratio=max(r['shape_gradient_ratio'] for r in items),
                             mean_coefficient=np.mean([r['shape_coefficient'] for r in items]),
                             gradient_cosine=np.mean([r['mean_var_gradient_cosine'] for r in items]),
                             tv_gradient_norm=np.mean([r['weighted_tv_q_gradient_norm'] for r in items]),
                             variance_gradient_norm=np.mean([r['var_q_gradient_norm'] for r in items])))
        summary.append(dict(experiment=arm,updates=400,actual_samples=3200,
                            mean_ratio=np.mean([r['shape_gradient_ratio'] for r in collected]),
                            maximum_ratio=max(r['shape_gradient_ratio'] for r in collected),
                            mean_cosine=np.mean([r['mean_var_gradient_cosine'] for r in collected]),
                            negative_cosine_fraction=np.mean([r['mean_var_gradient_cosine']<0 for r in collected])))
    csv_write(A/'gradient_by_step.csv',rows);csv_write(A/'gradient_summary.csv',summary)
    fig,axs=plt.subplots(2,2,figsize=(13,7),constrained_layout=True)
    for arm in ('e3','e3_mean005'):
        rr=[r for r in rows if r['experiment']==arm];steps=[r['step'] for r in rr]
        for ax,key,title in zip(axs.ravel(),['mean_ratio','max_ratio','gradient_cosine','tv_gradient_norm'],
                               ['Mean gradient ratio','Maximum per-sample ratio','Mean / variance gradient cosine','Weighted TV gradient norm at q']):
            ax.plot(steps,[r[key] for r in rr],label=arm);ax.set_title(title);ax.set_xlabel('Optimizer step');ax.grid(alpha=.25)
    axs[0,1].plot(range(1,401),[.05*min(s/50,1) for s in range(1,401)],'k--',label='Allowed bound')
    for ax in axs.ravel():ax.legend(fontsize=8)
    fig.savefig(A/'gradient_ratio_direction_tv.png',dpi=150);plt.close(fig)
    return summary

def repeat_tables_and_plots():
    rows=read('point_targets');lines=read('priority_lines');pairs=read('point_pairs')
    summary=[]
    for method in sorted(set(r['method'] for r in rows)):
        for repeat in (1,2,3):
            pp=[r for r in rows if r['method']==method and int(r['repeat'])==repeat and float(r['threshold'])==.1]
            ll=[r for r in lines if r['method']==method and int(r['repeat'])==repeat and float(r['threshold'])==.1]
            pair=[r for r in pairs if r['method']==method and int(r['repeat'])==repeat and float(r['threshold'])==.1]
            summary.append(dict(method=method,repeat=repeat,points=len(pp),localized_rate=rate(pp,'matched'),
                                local_depth_w1_um=mean(pp,'local_depth_w1_um'),pair_separated_rate=rate(pair,'separated'),
                                gapped_bridge_rate=rate([r for r in ll if float(r['gap_um'])>0],'bridged'),
                                continuous_false_gap_rate=rate([r for r in ll if float(r['gap_um'])==0],'false_gap')))
    csv_write(A/'per_repeat_summary.csv',summary)
    # Every fixed single-point target has its own depth-following curve for each
    # repeat and checkpoint role. No reconstruction averaging across acquisitions.
    fig,axs=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for ri,role in enumerate(('best','final')):
        for ci,repeat in enumerate((1,2,3)):
            ax=axs[ri,ci]
            for arm,color in zip(exp.ARMS,('tab:blue','tab:orange','tab:green')):
                for cell in range(1,5):
                    rr=[r for r in rows if r['method']==f'{arm}_{role}' and int(r['repeat'])==repeat and float(r['threshold'])==.1 and int(r['cell_id'])==cell]
                    rr.sort(key=lambda r:float(r['z_um']))
                    ax.plot([float(r['z_um']) for r in rr],[float(r['profile_peak_z_um']) for r in rr],
                            marker='o',markersize=3,lw=1,alpha=.6,color=color,label=arm if cell==1 else None)
            ax.plot([20,90],[20,90],'k--');ax.set(title=f'{role} · acquisition {repeat}',xlabel='True depth (µm)',ylabel='Local profile peak (µm)',ylim=(10,105));ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.savefig(A/'depth_following_three_repeats.png',dpi=150);plt.close(fig)
    return summary

def profile_figures():
    rows=read('fixed_profiles')
    figfiles=[]
    for sample,diagnostic,count,shape in [('T03','T03',12,(3,4)),('T04','T04',16,(4,4)),('T02','T02',13,(4,4))]:
        for role in ('best','final'):
            fig,axs=plt.subplots(*shape,figsize=(16,3.1*shape[0]),constrained_layout=True)
            for i,ax in enumerate(axs.ravel()):
                if i>=count:ax.axis('off');continue
                region=(f'{chr(65+i//4)}{i%4+1}' if sample=='T04' else str(i+1))
                selection=[r for r in rows if r['sample_id']==sample and int(r['subset'])==1 and r['region']==region]
                for method in ['mean_rl3','taylor_rl3']+[f'{arm}_{role}' for arm in exp.ARMS]:
                    rr=[r for r in selection if r['method']==method];rr.sort(key=lambda r:float(r['coordinate']))
                    x=np.array([float(r['coordinate']) for r in rr]);y=np.array([float(r['value']) for r in rr])
                    if len(y):ax.plot(x,y/max(y.max(),1e-30),marker='o' if sample!='T03' else None,ms=2,label=method)
                ref=[r for r in selection if r['method']==f'e3_{role}']
                ref.sort(key=lambda r:float(r['coordinate']))
                if ref and ref[0].get('gt_value'):
                    y=np.array([float(r['gt_value']) for r in ref]);ax.plot([float(r['coordinate']) for r in ref],y/max(y.max(),1e-30),'k--',label='GT')
                ax.set(title=f'{sample} region {region}',xlabel='Transverse pixel' if sample=='T03' else 'Native depth (µm)',ylim=(-.02,1.05));ax.grid(alpha=.25)
            axs.ravel()[0].legend(fontsize=6)
            path=A/f'{sample}_{role}_fixed_profiles.png';fig.savefig(path,dpi=150);plt.close(fig);figfiles.append(path)
    return figfiles

def artifact_audit():
    pre=json.loads((OUT/'preflight.json').read_text())
    for name,digest in pre['training_source_hashes'].items():assert old.sha256(name)==digest,name
    for name,digest in pre['config_hashes'].items():assert old.sha256(name)==digest,name
    assert old.sha256(OUT/'shared_initial_state.pt')==pre['shared_initial_file_sha256']
    assert old.sha256(pre['psf_path'])==pre['psf_sha256']
    indexed,fingerprint=load_dataset_index(exp.DATA);assert fingerprint==pre['dataset_fingerprint']
    sampler=FixedGlobalBatchScheduler(110,8,20260901)
    for i in range(400):sampler.next_batch()
    expected=sampler.state_dict()
    for arm in exp.ARMS:
        cp=torch.load(OUT/arm/'checkpoint_last.pt',map_location='cpu',weights_only=False)
        assert cp['completed_steps']==400 and cp['world_size']==5
        state=cp['scheduler_state']
        assert np.array_equal(state['order'],expected['order'])
        assert state['cursor']==expected['cursor'] and state['rng_state']==expected['rng_state']
        for group in cp['optimizer_state']['param_groups']:
            assert group['lr']==(1e-4 if group['group_name']=='network' else 1e-5)
        # Recompute the winning score over objects, with the first minimum as tie break.
        bystep=defaultdict(lambda:defaultdict(list))
        with (OUT/arm/'validation_metrics.csv').open() as f:
            for r in csv.DictReader(f):bystep[int(r['step'])][r['sample_id']].append(float(r['selection_score']))
        scores={s:np.mean([np.mean(v) for v in objs.values()]) for s,objs in bystep.items()}
        assert min(sorted(scores),key=scores.get)==cp['best_step']
    inventory=[]
    for arm in exp.ARMS:
        for role,cpname in [('best','checkpoint_best.pt'),('final','checkpoint_last.pt')]:
            digest=old.sha256(OUT/arm/cpname)
            for case in all_cases():
                d=OUT/'evaluation'/arm/role/case['id'];rec=json.loads((d/'complete.json').read_text())
                assert rec['complete'] and not rec['inference_target_or_gt_used']
                assert rec['checkpoint_sha256']==digest
                subset=case['path']/'subsets'/f"subset_{case['subset']:02d}.mat"
                assert old.sha256(subset)==rec['input_fingerprint']['subset_sha256']
                for filename,key in [('reconstruction.npy','prediction_sha256'),('anchor.npy','anchor_sha256')]:
                    path=d/filename;sha=old.sha256(path);assert sha==rec[key]
                    a=np.load(path);assert a.shape==(10,260,260) and a.dtype==np.float32 and np.isfinite(a).all() and np.min(a)>=0
                    inventory.append({'path':str(path.relative_to(OUT)),'sha256':sha})
    assert len(inventory)==1128
    csv_write(A/'prediction_anchor_sha256.csv',inventory)
    tests={}
    for sample,func in [('T03',t03),('T04',t04),('V03',v03)]:
        gt=_read_targets(DatasetItemKey(sample,1,'test',exp.DATA/sample),include_ground_truth=True)['ground_truth'][0]
        rows,_=func(gt,gt,{'method':'gt_identity'})
        rows=[r for r in rows if r['threshold_fraction']==.1]
        if sample=='T03':assert all(r['separated'] for r in rows)
        elif sample=='T04':assert all(r['separated'] for r in rows if r['separation_applicable'])
        else:assert all(r['detected'] for r in rows)
        tests[sample]={'GT_control_passed':True,'items':len(rows)}
    old.write_json(OUT/'local_evaluation_checks.json',tests)
    return {'prediction_and_anchor_hashes_verified':1128,'sampler_final_state_equal':True,
            'training_source_hashes_verified':len(pre['training_source_hashes']),
            'best_score_object_macro_recomputed':True,'geometry_GT_positive_controls':tests}

def enrich(acceptance):
    torch.set_num_threads(4)
    if not (OUT/'supplemental_gpu_acceptance.json').is_file():
        import os,subprocess
        from tools.run_v3_compare import free_gpus,inventory
        card=free_gpus('auto',required=1)[0]
        info=inventory()[card]
        assert not info['busy'] and info['memory']<=1024
        environment=os.environ.copy()
        environment['CUDA_VISIBLE_DEVICES']=info['uuid']
        environment['MPLCONFIGDIR']=str(OUT/'mpl_cache')
        subprocess.run([sys.executable,str(ROOT/'tools/v3_compare_supplemental_checks.py')],cwd=ROOT,env=environment,check=True)
    supplemental=json.loads((OUT/'supplemental_gpu_acceptance.json').read_text());assert supplemental['passed']
    gradients=gradient_audit();repeats=repeat_tables_and_plots();profile_files=profile_figures()
    audit=artifact_audit()
    from tools.v3_compare_report import _draw_projection_panel
    manifest=json.loads((A/'figure_manifest.json').read_text())
    # Include every original local scene separately, all three acquisitions.
    for case in all_cases():
        if case['split']!='priority':continue
        for role in ('best','final'):
            for display in ('shape','shared'):
                path=A/'actual_reconstruction_comparison'/case['id']/f'{role}_{display}.png'
                if path.exists():continue
                manifest.append(_draw_projection_panel(case,role,display,path))
    old.write_json(A/'figure_manifest.json',manifest)
    atlas=['# 实际重建图册','', '每个三维体只做一次显示归一化；shared 共用预测亮度上限，GT 自用显示尺度。三个重复独立展示，不平均预测。','']
    for case in all_cases():
        selected=[f for f in manifest if f['case_id']==case['id']]
        if not selected:continue
        atlas.append('## '+case['id']);atlas.append('')
        atlas.extend(f"- [{r['role']} · {r['display']}]({OUT/r['path']})" for r in selected);atlas.append('')
    (A/'actual_reconstruction_comparison/README_ZH.md').write_text('\n'.join(atlas))
    # Keep failed targets explicit, including misses and excess peaks, not only winners.
    failures=[]
    for table,condition in [('t03_lines',lambda r:not truthy(r['separated'])),
                            ('t04_axial',lambda r:truthy(r['separation_applicable']) and not truthy(r['separated'])),
                            ('v03_beads',lambda r:not truthy(r['detected']))]:
        failures.extend({'diagnostic':table,**r} for r in read(table) if float(r['threshold_fraction'])==.1 and condition(r))
    failures.extend({'diagnostic':'priority_lines',**r} for r in read('priority_lines') if float(r['threshold'])==.1 and (truthy(r['bridged']) or truthy(r['false_gap'])))
    failures.extend({'diagnostic':'priority_points',**r} for r in read('point_targets') if float(r['threshold'])==.1 and not truthy(r['matched']))
    failures.extend({'diagnostic':'priority_axial',**r} for r in read('axial_points') if float(r['threshold'])==.1 and not truthy(r['matched']))
    csv_write(A/'failure_cases.csv',failures)
    zeros=[r for r in read('metrics') if r['split']=='zero'];csv_write(A/'zero_input.csv',zeros)
    main=(OUT/'REPORT_ZH.md').read_text()
    extra=['','## 按对象看 C 相对 B 的取舍','','| final 测试对象 | B 形状误差 ↓ | C 形状误差 ↓ | B 深度分布误差 µm ↓ | C 深度分布误差 µm ↓ |','|---|---:|---:|---:|---:|']
    objects=read('per_object')
    for sample in ('T02','T03','T04'):
        b=next(r for r in objects if r['method']=='e3_final' and r['sample_id']==sample)
        c=next(r for r in objects if r['method']=='e3_mean005_final' and r['sample_id']==sample)
        extra.append(f"| {sample} | {float(b['gt_scale_aligned_nrmse']):.4f} | {float(c['gt_scale_aligned_nrmse']):.4f} | {float(b['gt_axial_w1_um']):.2f} | {float(c['gt_axial_w1_um']):.2f} |")
    extra += ['', 'C 的平均形状误差略好，是 T02、T04 的改善抵消了 T03 的退步。它并没有在所有结构上都比 B 更好。',
              '', 'T03 的三线分开率只从 79/120 增到 80/120，多了一个“线组×子集”成功。与此同时，中央线段出现断裂的线组从 0/120 增到 12/120。十个子集彼此重叠，这一个成功差异不能当成稳定优势。',
              '', 'T04 是沿 Z 重叠的双层细线。B 和 C 在 90 个 20/30/40 µm 区域×子集组合中都未通过双峰分离。C 的能量分布更接近 GT，并不等同于两个真实层都被独立找回。',
              '', 'V03 是验证集，单独作诊断。按填充微球的强度质心匹配，B/C 的检出率约为 91.8%/93.3%，但匹配成功微球的平均 Z 误差约为 1.80/3.04 µm。微球是有体积的实心球，单个最亮像素可能偏离球心，因此采用连通域质心；合并成同一个连通域只能匹配一个微球。',
              '', '旧诊断场景也没有显示 C 稳定获胜：final 单点局部深度分布误差 B/C 约为 2.57/5.13 µm；断口误连接率约为 22.2%/24.4%。三次独立采集的逐次结果保留在下表和图册中。',
              '', '| 方法 final | 重复 | 点定位率 | 局部深度误差 µm | 点对分开率 | 断口误连接率 |','|---|---:|---:|---:|---:|---:|']
    for r in repeats:
        if r['method'].endswith('_final'):
            extra.append(f"| {r['method']} | {r['repeat']} | {r['localized_rate']:.3f} | {r['local_depth_w1_um']:.2f} | {r['pair_separated_rate']:.3f} | {r['gapped_bridge_rate']:.3f} |")
    extra+=['','## 亮度和梯度的实际核对','','E3 的输出密度单位和 GT 原始密度单位不同；没有把正式预测按 GT 重新拟合亮度。aligned NRMSE 只在计算形状误差时消除一个整体尺度。原始 XY-SSIM 会受亮度影响，不能单独拿它判断结构优劣。',
            '', '共同前向模型下，final 的归一化均值误差 A/B/C 约为 13.1006/0.1036/0.1113；绝对 log 方差误差约为 0.4267/0.7657/0.8634。E3 明显改善均值亮度拟合，但没有同时降低绝对方差误差；这是两个统计模型/尺度仍有冲突的证据，不能说 loss 问题已经全部解决。',
            '', 'B/C 每组 3,200 个训练样本的日志已逐条与相同抽样器的顺序核对。C 在每个样本 q 处都满足渐增后的 5% 上限；这并不保证 Adam 参数更新也只改变 5%。普通 E3 mean 对 q 的梯度为 None，只影响亮度参数；方差对 gamma 的梯度为 None。',
            '', '| 组别 | 实际平均 mean/var 梯度比 | 实际最大比 | 两梯度方向平均余弦 | 方向相反样本占比 |','|---|---:|---:|---:|---:|']
    for r in gradients:extra.append(f"| {r['experiment']} | {r['mean_ratio']:.5f} | {r['maximum_ratio']:.8f} | {r['mean_cosine']:.3f} | {r['negative_cosine_fraction']:.3f} |")
    extra+=['','零输入输出：','','| 模型 | 输出最大值 | 输出总量 |','|---|---:|---:|']
    for r in zeros:extra.append(f"| {r['method']} | {float(r['prediction_max']):.6g} | {float(r['prediction_sum']):.6g} |")
    extra+=['','B/C 的全零输出来自 E3 亮度尺度的显式约束：十帧均值为零时，解析增益也为零。这是已验证的边界行为，不能单凭它断言非零输入的背景也都正确。','## 具体建议','','本轮更适合保留 B 作为下一轮参照。C 不作为全面替代：它对 T02 的形状改善值得保留，但当前从头启用 5% 约束带来了 T03 假断口、单点深度和背景方面的代价。',
            '', '下一轮若继续研究 mean 参与结构，优先比较“先用 E3 把结构学稳，再晚些启用较小 mean 梯度”的方案，同时保留原 B 对照；不要只继续增大 mean 权重。T04 双层失败应单独排查物理模型可辨识性与训练分布，不能靠调亮显示或插值制造两层。这个建议未自动执行。',
            '', '本轮只有一个训练种子，不能把极小均值差当成统计显著。训练、验证和测试始终按新分组；P07/P09/P11 不参与正式测试均值。',
            '', '## 完整交付入口','',f'- [全部实际重建图册]({A}/actual_reconstruction_comparison/README_ZH.md)',
            f'- [三次采集深度跟随图]({A}/depth_following_three_repeats.png)',
            f'- [逐步梯度、方向和 TV 曲线]({A}/gradient_ratio_direction_tv.png)',
            f'- [全部固定剖面 CSV]({A}/fixed_profiles.csv)',f'- [失败案例 CSV]({A}/failure_cases.csv)',
            f'- [预测和 anchor 哈希清单]({A}/prediction_anchor_sha256.csv)']
    extra.extend(f'- [{p.stem}]({p})' for p in profile_files)
    (OUT/'REPORT_ZH.md').write_text(main+'\n'+'\n'.join(extra)+'\n')
    # Save standalone, reviewable evaluation rules and their source fingerprints.
    rules={'thresholds':list((.05,.1,.2)),'primary_threshold':.1,'xy_tolerance_pixels':2,'z_tolerance_um':10,
           't03':'central 60%; independent 1D peaks; one-to-one centers within 2 pixels; selected native layer within 10um; both valley/weak<=.8',
           't04':'native 10-layer ROI sum; one-to-one peaks; 10um separation N/A; eligibility uses full-volume max',
           'v03':'filled-sphere component intensity centroids; one-to-one; merged component cannot match two beads',
           'repeat_policy':'three independent acquisitions displayed separately; ten overlapping subsets not independent acquisitions',
           'frozen_training_sources':str(OUT/'preflight.json'),'trained_checkpoints_unchanged':True}
    old.write_json(OUT/'evaluation_rules.json',rules)
    for p in [Path(__file__),ROOT/'tools/v3_compare_local_audit.py',ROOT/'tools/v3_compare_evaluation.py',ROOT/'tools/v3_compare_report.py',ROOT/'tools/v3_compare_supplemental_checks.py']:
        dest=OUT/'evaluation_source_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    acceptance['supplemental_gpu_checks']=supplemental
    acceptance['full_artifact_audit']=audit
    acceptance['all_priority_acquisitions_visualized']=33
    acceptance['comparison_panels']=len(manifest)
    acceptance['source_sha256'].update({str(p.relative_to(ROOT)):old.sha256(p) for p in [Path(__file__),ROOT/'tools/v3_compare_local_audit.py',ROOT/'tools/v3_compare_report.py']})
    acceptance['artifact_sha256'].update({str(p.relative_to(OUT)):old.sha256(p) for p in [OUT/'REPORT_ZH.md',OUT/'evaluation_rules.json',A/'figure_manifest.json']})
    old.write_json(OUT/'final_acceptance.json',acceptance)
    old.write_json(OUT/'complete.json',{'complete':True,'passed':True,'primary_predictions':564,'physical_anchors':564,
                                      'report_sha256':old.sha256(OUT/'REPORT_ZH.md'),'acceptance_sha256':old.sha256(OUT/'final_acceptance.json')})

if __name__=='__main__':enrich(json.loads((OUT/'final_acceptance.json').read_text()))
