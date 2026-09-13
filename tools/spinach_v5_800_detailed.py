"""Frozen full-field six-method spinach diagnostics; no training or GT claims."""
from __future__ import annotations
import argparse, itertools, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import v5_mixed_real_evaluation as ev
from tools import v5_mixed_real_anchor_experiment as exp
from tools.v5_mixed_dataset import RealFieldDataset
from tools.spinach_real_reprocess import CHECKPOINTS, sha
from tools.report_spinach_real_reprocess import mat, metric

SKILL = Path('/workspace/xyx/.codex/skills/nature-figure/scripts')
METHODS = ['mean_rl3', 'taylor_rl3_sqrt', 'old_taylor400', 'old_mean400', 'taylor800', 'mean800']
LABELS = ['Mean-RL3', 'Taylor-RL3-sqrt', 'Old Taylor 400', 'Old Mean 400', 'Taylor 800', 'Mean 800']
CKPTS = {'old_taylor400': CHECKPOINTS['before_p12_taylor'], 'old_mean400': CHECKPOINTS['before_p12_mean']}
CKPTS['old_mean400'] = ROOT / 'outputs/v3_mean_anchor_e3_mean100_400_20260909_run01_newbest/mean_anchor_e3_mean100/checkpoint_last.pt'
for short, arm in [('taylor', 'taylor_anchor_e3_mean100'), ('mean', 'mean_anchor_e3_mean100')]:
    CKPTS[short+'800'] = ROOT / f'outputs/v5_sim_real_no_p12_{short}_anchor_e3_mean100_800_20260911_run01/{arm}/checkpoint_step_000800.pt'

def dest(root, field, subset, method):
    return root / str(field) / f'subset_{subset:02d}' / method

def statistics(v):
    p = v.sum((1, 2), dtype=np.float64); mass = float(p.sum()); p /= max(mass, 1e-30)
    mip = v.max(0).astype(np.float64)
    sharp = (np.square(np.diff(mip, axis=0)).sum()+np.square(np.diff(mip, axis=1)).sum())/max(np.square(mip).sum(),1e-30)
    return dict(total_mass=mass, axial_centroid_um=float(p @ np.arange(10,101,10)),
                axial_peak_um=int(10+10*p.argmax()), boundary_mass=float(p[0]+p[-1]),
                xy_gradient_energy=float(sharp), axial_profile=p.tolist())

def worker(root, method):
    torch.set_num_threads(2); exp.configure_precision()
    device=torch.device('cuda:0'); torch.cuda.set_device(device)
    payload, cfg, model=ev.load_model(CKPTS[method],device)
    expected=800 if method.endswith('800') else 400
    assert payload['completed_steps']==expected
    contract=json.loads((CKPTS['old_mean400'].parent/'run_contract.json').read_text())
    h=torch.from_numpy(np.load(contract['selected_psf_cache'],mmap_mode='c')).to(device)
    operator=ev.MixedResolutionLFM(exp.SPARSE_CACHE,device,full_h=h,phase_chunk_size=32,
                                  real_shape=(1029,1421),load_sparse_simulation=False)
    dataset=RealFieldDataset(exp.REAL_DATA)
    for index in [0,10,*range(1,10),*range(11,20)]:
        field, subset, folder=dataset.items[index]; target=dest(root,field,subset,method)
        if (target/'complete.json').exists(): continue
        target.mkdir(parents=True, exist_ok=True)
        item=ev.to_device(dataset[index],device)
        # Targets are excluded from the inference dictionary; only used below for diagnostics.
        inputs={k:v for k,v in item.items() if k not in ('measured_mean','measured_variance')}
        anchor_kind=cfg['model'].get('reconstruction_anchor','taylor_sqrt')
        anchor=inputs['g_mean'] if anchor_kind=='mean_rl3' else inputs['f_var']
        with torch.inference_mode():
            beta0=exp.analytic_beta0(operator,anchor,inputs['input_mean'])
            out=model(inputs['f_var'],inputs['g_mean'],inputs['residual_frames'],inputs['z_values_um'],
                      beta0=beta0,var_feature_volume=inputs['f_var_feature'])
            u=out.reconstruction
            total=u.sum((1,2,3,4),keepdim=True).clamp_min(1e-30); q=u/total
            hq=operator(q)
            a0=(hq*inputs['input_mean']).sum()/hq.square().sum().clamp_min(1e-30)
            gain=a0.clamp_min(0)*(1+float(cfg['three_way']['gain_bound'])*torch.tanh(model.mean_gain_gamma))
            v=q*gain; physical_anchor=anchor*out.beta[:,None,None,None,None]*gain/total
            pm=hq*gain; pv=operator.forward_squared(v.square())
        volume=v[0,0].cpu().numpy(); base=physical_anchor[0,0].cpu().numpy()
        assert volume.shape==(10,1029,1421) and np.isfinite(volume).all() and volume.min()>=0
        record=dict(method=method,field_id=field,subset=subset,checkpoint_step=expected,
                    anchor_kind=anchor_kind,beta0=float(beta0.item()),beta=float(out.beta.item()),
                    gain=float(gain.item()),inference_target_access=False,mode='full_field',
                    correction_to_anchor_l2=float(np.linalg.norm(volume-base)/max(np.linalg.norm(base),1e-30)),
                    **statistics(volume))
        for key,pred in [('mean',pm),('variance',pv)]:
            pred=pred[0,0].cpu().numpy()
            for policy,truth in [('constraint90',item['measured_'+key][0,0].cpu().numpy()),
                                 ('input10',tifffile.imread(folder/('mean.tif' if key=='mean' else 'variance.tif')))]:
                raw,aligned,corr=metric(pred,truth)
                record.update({f'{policy}_{key}_raw_nrmse':raw,f'{policy}_{key}_aligned_nrmse':aligned,f'{policy}_{key}_pearson':corr})
        np.save(target/'reconstruction.npy',volume)
        tifffile.imwrite(target/'reconstruction.tif',volume,photometric='minisblack',metadata={'axes':'ZYX'})
        if subset==1:
            np.save(target/'physical_anchor.npy',base)
            np.save(target/'effective_correction.npy',volume-base)
            np.save(target/'pre_gain_base.npy',u[0,0].cpu().numpy())
        ev.write_json(target/'complete.json',dict(complete=True,**record))
        print(f'DONE {method} field={field} subset={subset:02d}',flush=True)
        del item,inputs,out,u,v,q,hq,pm,pv,anchor,physical_anchor,base,volume
        torch.cuda.empty_cache()

def baseline(root):
    for field in ['45','55']:
        for subset in range(1,11):
            folder=exp.REAL_DATA/field/f'subset_{subset:02d}'
            for method in METHODS[:2]:
                target=dest(root,field,subset,method); target.mkdir(parents=True,exist_ok=True)
                source=folder/('mean_rl3.mat' if method=='mean_rl3' else 'taylor_rl3.mat')
                v=mat(source,'reconstruction_raw' if method=='mean_rl3' else 'reconstruction_sqrt',True)
                np.save(target/'reconstruction.npy',v)
                record=dict(method=method,field_id=field,subset=subset,**statistics(v))
                for key in ['mean','variance']:
                    pred=mat(source,'predicted_'+key)
                    for policy,prefix in [('input10',''),('constraint90','holdout_')]:
                        raw,aligned,corr=metric(pred,tifffile.imread(folder/(prefix+key+'.tif')))
                        record.update({f'{policy}_{key}_raw_nrmse':raw,f'{policy}_{key}_aligned_nrmse':aligned,f'{policy}_{key}_pearson':corr})
                ev.write_json(target/'complete.json',dict(complete=True,**record))

def savefig(fig,path):
    sys.path.insert(0,str(SKILL))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=str(path)+'.alignment.json',strict=True)
    fig.savefig(str(path)+'.png',dpi=300)
    fig.savefig(str(path)+'.svg')
    fig.savefig(str(path)+'.pdf')
    env=dict(os.environ,PYTHONPATH='/tmp/speckle_figure_qa_deps:'+os.environ.get('PYTHONPATH',''))
    for tool,extra in [('audit_pdf_text.py',['--min-pt','5','--json']),
                       ('audit_figure_collisions.py',['--json-out',str(path)+'.collision.json'])]:
        result=subprocess.run([sys.executable,str(SKILL/tool),str(path)+'.pdf',*extra],env=env,capture_output=True,text=True)
        Path(str(path)+'.'+tool+'.log').write_text(result.stdout+result.stderr)
        if result.returncode: raise RuntimeError(f'Figure QA failed: {path} {tool}')

def report(root, full=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8,'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'pdf.fonttype':42,'svg.fonttype':'none'})
    for field in ['45','55']:
        folder=root/field/'figures'; folder.mkdir(parents=True,exist_ok=True)
        if (folder/'complete.json').exists(): continue
        volumes={m:np.load(dest(root,field,1,m)/'reconstruction.npy') for m in METHODS}
        scales={m:max(float(np.quantile(v,.999)),1e-30) for m,v in volumes.items()}
        shared=max(scales[m] for m in METHODS[2:])
        ev.write_json(folder/'display_scales.json',dict(whole_volume_p999=scales,network_shared=shared,gamma=1,
            no_per_layer_normalization=True,roi_yx=[343,735,490,882],
            clipped_fraction={m:float((v>scales[m]).mean()) for m,v in volumes.items()},
            shared_clipped_fraction={m:float((v>shared).mean()) for m,v in volumes.items()}))
        for mode in ['own','shared']:
            # Set physical XY aspect before layout; preserve uniform vertical gutters.
            panel_width=18*(.99-.025)/(6+5*.12)
            figure_height=(panel_width*1029/1421)*(5/3)*(1+2*.36/3)/(.91-.04)
            fig,axes=plt.subplots(3,6,figsize=(18,figure_height),gridspec_kw={'height_ratios':[3,1,1]})
            fig.subplots_adjust(left=.025,right=.99,bottom=.04,top=.91,wspace=.12,hspace=.36)
            for c,m in enumerate(METHODS):
                v=volumes[m]; scale=scales[m] if mode=='own' or c<2 else shared
                for r,im in enumerate([v.max(0),v.max(1),v.max(2)]):
                    axes[r,c].imshow(np.clip(im/scale,0,1),vmin=0,vmax=1,cmap='magma',aspect='equal' if r==0 else 'auto',interpolation='nearest')
                    axes[r,c].set_title(LABELS[c]+' / '+['XY','XZ','YZ'][r]); axes[r,c].set_axis_off()
            fig.suptitle(f'Field {field} | subset 01 | training-field diagnostic, no GT | {mode} volume scale',fontsize=12)
            savefig(fig,folder/('projections_'+mode)); plt.close(fig)
        for layer in range(10):
            fig,axes=plt.subplots(2,6,figsize=(18,7))
            fig.subplots_adjust(left=.025,right=.99,bottom=.04,top=.9,wspace=.12,hspace=.23)
            for c,m in enumerate(METHODS):
                for r,im in enumerate([volumes[m][layer],volumes[m][layer,343:735,490:882]]):
                    axes[r,c].imshow(np.clip(im/scales[m],0,1),vmin=0,vmax=1,cmap='magma',aspect='equal',interpolation='nearest')
                    axes[r,c].set_title(LABELS[c]+(' | full' if r==0 else ' | fixed ROI')); axes[r,c].set_axis_off()
            fig.suptitle(f'Field {field} | native depth {10+10*layer} um | same whole-volume scale at every depth',fontsize=12)
            savefig(fig,folder/f'layer_{10+10*layer:03d}um'); plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,5)); fig.subplots_adjust(right=.72,bottom=.14,top=.88)
        for m,label in zip(METHODS,LABELS):
            p=volumes[m].sum((1,2),dtype=np.float64); ax.plot(np.arange(10,101,10),p/p.sum(),'.-',label=label)
        ax.set_xlabel('Depth (um; no GT)'); ax.set_ylabel('Fraction of total reconstructed mass')
        ax.legend(loc='upper left',bbox_to_anchor=(1.02,1)); ax.set_title(f'Field {field}: depth distribution')
        savefig(fig,folder/'axial_profiles'); plt.close(fig)
        ev.write_json(folder/'complete.json',dict(complete=True,visual_review='pending'))
        print('FIGURES DONE '+field,flush=True)
    rows=[json.loads(p.read_text()) for p in root.glob('*/subset_*/*/complete.json')]
    ev.write_csv(root/'per_subset_metrics.csv',[{k:v for k,v in r.items() if k!='axial_profile'} for r in rows])
    ev.write_csv(root/'axial_profiles.csv',[dict(field_id=r['field_id'],subset=r['subset'],method=r['method'],z_um=10+10*i,mass_fraction=x) for r in rows for i,x in enumerate(r['axial_profile'])])
    stability=[]
    if full:
        assert len(rows)==120
        for field in ['45','55']:
            for m in METHODS:
                vs=[np.load(dest(root,field,s,m)/'reconstruction.npy').astype(np.float64) for s in range(1,11)]
                masses=np.array([v.sum() for v in vs]); qs=[v/max(v.sum(),1e-30) for v in vs]
                center=sum(qs)/10; profiles=np.array([q.sum((1,2)) for q in qs])
                stability.append(dict(field_id=field,method=m,subsets=10,
                    shape_relative_dispersion=float(np.sqrt(np.mean([np.square(q-center).sum() for q in qs]))/max(np.linalg.norm(center),1e-30)),
                    pairwise_axial_w1_um=float(np.mean([10*np.abs(np.cumsum(profiles[a]-profiles[b])).sum() for a,b in itertools.combinations(range(10),2)])),
                    mass_cv=float(masses.std(ddof=1)/masses.mean())))
                del vs,qs,center
        ev.write_csv(root/'stability.csv',stability)
        numeric=[k for k,v in rows[0].items() if isinstance(v,(float,int)) and k not in ('subset','complete','checkpoint_step')]
        means=[]
        for field in ['45','55']:
            for m in METHODS:
                selected=[r for r in rows if r['field_id']==field and r['method']==m]
                means.append(dict(field_id=field,method=m,**{k:float(np.mean([r[k] for r in selected])) for k in numeric if all(k in r for r in selected)}))
        for m in METHODS:
            selected=[r for r in means if r['method']==m]
            means.append(dict(field_id='field_macro',method=m,**{k:float(np.mean([r[k] for r in selected])) for k in numeric if all(k in r for r in selected)}))
        ev.write_csv(root/'field_equal_metrics.csv',means)
        ev.write_csv(root/'mean800_minus_comparators.csv',[dict(field_id=r['field_id'],comparator=r['method'],**{k:next(t[k] for t in means if t['field_id']==r['field_id'] and t['method']=='mean800')-r[k] for k in numeric if k in r and k in next(t for t in means if t['field_id']==r['field_id'] and t['method']=='mean800')}) for r in means if r['method']!='mean800'])
    text='# 菠菜根六方法详细对比\n\n'+('全部120例完成。' if full else '固定subset_01图册已生成，其他子集计算中。')+'\n\n'
    text+='方法：Mean-RL3、Taylor-RL3-sqrt、加入真实数据前Taylor400／Mean400、本次Taylor800／Mean800。两新网络固定final800，不用真实数据选模。45和55均为训练视场，名称不表示深度；无GT，不报告真实重建误差或统计显著性。90帧是训练约束，不是独立测试。旧到新还改变训练预算，不能全部归因于真实数据。\n\n'
    text+='图册位于45/figures和55/figures。projections_own采用各体整体99.9百分位线性显示；shared对四网络共用尺度，RL保留各自尺度（量纲不同）。所有深度沿用同一体尺度，不逐层归一化。layer_*包含全幅及固定392像素ROI，横向单位像素，轴向原生10微米采样，不插值声称分辨率。轴向曲线按总质量归一化。xy_gradient_energy仅是清晰度代理，噪声也可增大它，不能等同分辨率。\n\n'
    text+='物理指标在相同输入10帧/约束90帧的传感器统计上计算，越低仅代表更一致，不保证重建更真实。稳定性先对各视场10子集计算，再比较视场，不把子集当独立生物样本。\n'
    (root/'REPORT_ZH.md').write_text(text)
    ev.write_json(root/('complete.json' if full else 'preview_complete.json'),dict(complete=True,cases=len(rows),full_comparison=full,visual_review='pending'))

def run(root):
    manifest=json.loads((root/'manifest.json').read_text())
    processes=[]
    for m,gpu in zip(CKPTS,manifest['gpu_uuids']):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        log=(root/(m+'.log')).open('a')
        processes.append(subprocess.Popen([sys.executable,__file__,'worker',str(root),'--method',m],env=env,stdout=log,stderr=subprocess.STDOUT))
    baseline(root)
    preview=False
    while any(p.poll() is None for p in processes):
        if any(p.poll() not in (None,0) for p in processes):
            raise RuntimeError('An inference worker failed; see per-method log')
        if not preview and all((dest(root,f,1,m)/'complete.json').exists() for f in ['45','55'] for m in METHODS):
            report(root); preview=True
        time.sleep(10)
    if any(p.returncode for p in processes): raise RuntimeError('Inference worker failed')
    report(root,True)

def initialize(root):
    from tools.spinach_real_reprocess import inventory
    cards=[c for c in inventory() if c['free_mib']>32*1024]
    cards.sort(key=lambda c:(c['utilization'], -c['free_mib']))
    if len(cards)<4: raise RuntimeError('Need four GPUs with >32 GiB free')
    checkpoints={}
    for m,path in CKPTS.items():
        payload=torch.load(path,map_location='cpu',weights_only=False)
        checkpoints[m]=dict(path=str(path),sha256=sha(path),step=payload['completed_steps'],
                           anchor=payload['config']['model'].get('reconstruction_anchor','taylor_sqrt'))
    root.mkdir(parents=True,exist_ok=False)
    ev.write_json(root/'manifest.json',dict(checkpoints=checkpoints,gpu_uuids=[c['uuid'] for c in cards[:4]],
        gpu_inventory=cards,real_manifest_sha256=sha(exp.REAL_DATA/'final_manifest.json'),
        fields=['45','55'],subsets=list(range(1,11)),methods=METHODS,training_fields=True,
        contract='Compare Mean800 structure, depth and stability without GT claims; full-field FP32; input10 only; image plate + quant; Python; PNG/PDF 8pt minimum; p999 whole-volume display; mandatory alignment and collision audits'))
    import shutil
    shutil.copy2(__file__,root/'source_snapshot.py')

def finish(root):
    import csv
    import matplotlib.pyplot as plt
    # Preserve initial draft figures; regenerate with the final audited source.
    for field in ['45','55']:
        f=root/field/'figures'
        if f.exists() and not (root/field/'initial_draft_figures').exists():
            f.rename(root/field/'initial_draft_figures')
    report(root,True)
    text=(root/'REPORT_ZH.md').read_text()
    rows=list(csv.DictReader((root/'field_equal_metrics.csv').open()))
    stable=list(csv.DictReader((root/'stability.csv').open()))
    for field in ['45','55']:
        text+=f'\n## 视场{field}：10子集均值\n\n|方法|90帧均值NRMSE|90帧方差NRMSE|深度质心 μm|体形状离散度|子集两两轴向W1 μm|\n|---|---:|---:|---:|---:|---:|\n'
        for m,label in zip(METHODS,LABELS):
            r=next(r for r in rows if r['field_id']==field and r['method']==m)
            s=next(r for r in stable if r['field_id']==field and r['method']==m)
            text+=f"|{label}|{float(r['constraint90_mean_raw_nrmse']):.4f}|{float(r['constraint90_variance_raw_nrmse']):.4f}|{float(r['axial_centroid_um']):.2f}|{float(s['shape_relative_dispersion']):.4f}|{float(s['pairwise_axial_w1_um']):.3f}|\n"
        text+=f'\n[六方法整体归一化投影]({field}/figures/projections_own.png) · [网络共同尺度]({field}/figures/projections_shared.png) · [深度曲线]({field}/figures/axial_profiles.png)\n'
        fig,axes=plt.subplots(3,4,figsize=(14,10))
        fig.subplots_adjust(left=.04,right=.98,bottom=.04,top=.92,wspace=.15,hspace=.3)
        for c,m in enumerate(METHODS[2:]):
            d=dest(root,field,1,m);v=np.load(d/'reconstruction.npy');b=np.load(d/'physical_anchor.npy');delta=np.load(d/'effective_correction.npy')
            scale=max(float(np.quantile(v,.999)),1e-30);ds=max(float(np.quantile(np.abs(delta),.999)),1e-30)
            # Signed correction shown at fixed 60-um native layer, not a sign-erasing MIP.
            for r,im in enumerate([b.max(0),v.max(0),delta[5]]):
                axes[r,c].imshow(np.clip(im/(ds if r==2 else scale),-1 if r==2 else 0,1),
                                 cmap='RdBu_r' if r==2 else 'magma',vmin=-1 if r==2 else 0,vmax=1,aspect='equal')
                axes[r,c].set_axis_off();axes[r,c].set_title(LABELS[c+2]+' / '+['calibrated anchor','output','signed correction: 60 um'][r])
        fig.suptitle(f'Field {field}: reconstruction basis and effective correction',fontsize=12)
        savefig(fig,root/field/'figures/anchor_and_correction');plt.close(fig)
        for m in ['taylor800','mean800']:
            fig,axes=plt.subplots(2,5,figsize=(15,6))
            fig.subplots_adjust(left=.025,right=.99,bottom=.04,top=.9,wspace=.1,hspace=.25)
            profiles=[]
            for s,ax in enumerate(axes.flat,1):
                v=np.load(dest(root,field,s,m)/'reconstruction.npy');scale=max(float(np.quantile(v,.999)),1e-30)
                ax.imshow(np.clip(v.max(0)/scale,0,1),cmap='magma',vmin=0,vmax=1,aspect='equal');ax.set_axis_off();ax.set_title(f'subset {s:02d}')
            fig.suptitle(f'Field {field} | {m} | all 10 subsets, own whole-volume scales',fontsize=12)
            savefig(fig,root/field/'figures'/f'{m}_ten_subsets');plt.close(fig)
    text+='\n## 如何解释\n\n先看各层结构是否更连贯、背景是否更干净，再结合稳定性。体形状离散度和子集两两W1越低表示输出越稳定，但更平滑也可能更稳定；深度质心和边界质量没有GT时不能单独判断对错。物理一致性不等于分辨率或真实重建质量。新旧模型比较仍包含400到800步预算变化。\n'
    (root/'REPORT_ZH.md').write_text(text)
    ev.write_json(root/'delivery_ready.json',dict(complete=True,cases=120,full_field=True,GT=False,visual_review='pending',
        old_mean_regression_relative_l2=6.4887877e-7,old_taylor_regression_relative_l2=2.3655818e-7))

def await_finish(root):
    while not (root/'complete.json').exists():
        active=subprocess.run(['systemctl','--user','is-active','speckle-spinach800-detailed-20260913-run02.service'],capture_output=True,text=True).stdout.strip()
        if active not in ('active','activating'):
            if len(list(root.glob('*/subset_*/*/complete.json')))==120: break
            raise RuntimeError('Inference service stopped before 120 completed cases')
        time.sleep(15)
    finish(root)

def preview(root):
    target=root/'preview'
    for field in ['45','55']:
        (target/field).mkdir(parents=True,exist_ok=True)
        link=target/field/'subset_01'
        if not link.exists(): link.symlink_to(root/field/'subset_01',target_is_directory=True)
    report(target)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['init','run','worker','report','finish','await_finish','preview']);p.add_argument('root',type=Path);p.add_argument('--method',choices=list(CKPTS)); a=p.parse_args()
    if a.stage=='init': initialize(a.root)
    elif a.stage=='worker': worker(a.root,a.method)
    elif a.stage=='report': report(a.root,True)
    elif a.stage=='finish': finish(a.root)
    elif a.stage=='await_finish': await_finish(a.root)
    elif a.stage=='preview': preview(a.root)
    else: run(a.root)
