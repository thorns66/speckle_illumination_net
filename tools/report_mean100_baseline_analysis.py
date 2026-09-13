"""CPU report for frozen Mean100 analysis. No new training or reconstruction."""
from __future__ import annotations
import argparse
from collections import defaultdict
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mean100_analysis_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
from tools import mean100_baseline_analysis as core
from tools import v3_compare_report as legacy

SKILL = Path('/workspace/xyx/.codex/skills/nature-figure/scripts')
sys.path.insert(0, str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment
PRIMARY, METHODS, LABELS = core.PRIMARY, core.METHODS, dict(core.LABELS, GT='Ground truth')
COLORS = dict(zip(PRIMARY, ['#7b8794', '#d49a52', '#8173b5', '#197ca5']))
COLORS.update(mean100_best740='#5ca4b0', GT='#202020')
METRICS = ['gt_raw_nrmse', 'gt_scale_aligned_nrmse', 'gt_axial_w1_um', 'gt_axial_mass_l1',
           'gt_support_outside_pm10_mass', 'gt_xy_mip_ssim', 'background_xy_mass_fraction',
           'common_normalized_mean_loss', 'common_absolute_log_variance_loss',
           'common_normalized_shape_variance_loss', 'common_e3_score', 'prediction_sum',
           'mass_ratio_to_truth', 'correction_to_anchor_l2']
GROUPS = {'A_RL_vs_Mean100': ('GT', 'mean_rl3', 'taylor_rl3_sqrt', 'mean100_final800'),
          'B_Taylor100_vs_Mean100': ('GT', 'taylor100_final800', 'mean100_final800')}
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                     'font.size': 11, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                     'axes.spines.top': False, 'axes.spines.right': False})


def number(r, key):
    try: return float(r[key])
    except (ValueError, KeyError, TypeError): return float('nan')


def avg(rows, key):
    v = [number(r, key) for r in rows]; v = [x for x in v if math.isfinite(x)]
    return float(np.mean(v)) if v else float('nan')


def groups(rows, keys):
    result = defaultdict(list)
    for row in rows: result[tuple(row[k] for k in keys)].append(row)
    return result


def aggregate(rows, keys, metrics):
    return [{**dict(zip(keys, key)), 'n': len(items), **{m: avg(items, m) for m in metrics}}
            for key, items in sorted(groups(rows, keys).items())]


def paired(rows, keys, metrics, baseline='mean100_final800'):
    output = []
    for key, items in groups(rows, keys).items():
        lookup = {r['method']: r for r in items}
        if baseline not in lookup: continue
        for method, r in lookup.items():
            if method == baseline: continue
            output.append({**dict(zip(keys, key)), 'baseline': baseline, 'comparator': method,
                           **{m: number(lookup[baseline], m)-number(r, m) for m in metrics}})
    return output


def stability_values(values):
    values = np.asarray(values, dtype=np.float64)
    masses = values.sum((1, 2, 3))
    assert (masses > 0).all()
    q = values / masses[:, None, None, None]
    center = q.mean(0); profiles = q.sum((2, 3))
    pairwise = [float(np.abs(np.cumsum(profiles[a]-profiles[b])).sum()*10)
                for a, b in itertools.combinations(range(len(q)), 2)]
    return dict(shape_relative_dispersion=float(np.sqrt(np.mean(np.sum((q-center)**2, axis=(1,2,3))))/np.linalg.norm(center)),
                pairwise_axial_w1_um=float(np.mean(pairwise)),
                mass_cv=float(masses.std(ddof=1)/masses.mean()), pairs=len(pairwise), subsets=len(q))


def summarize(root, manifest, cases):
    a = root / 'analysis'
    rows = core.read_csv(a / 'metrics.csv')
    assert len(rows) == 470 and all(len(v) == 94 for v in groups(rows, ('method',)).values())
    official = [r for r in rows if r['split'] in ('test', 'validation')]
    obj = aggregate(official, ('method', 'split', 'sample_id'), METRICS)
    for row in obj:
        items=[r for r in official if all(r[k]==row[k] for k in ('method','split','sample_id'))]
        for key in ('gt_scale_aligned_nrmse','gt_axial_w1_um','background_xy_mass_fraction'):
            row[key+'_subset_sd']=float(np.std([number(r,key) for r in items],ddof=1))
    summary = aggregate(obj, ('method', 'split'), METRICS)
    core.write_csv(a / 'per_object.csv', obj)
    core.write_csv(a / 'summary.csv', summary)
    core.write_csv(a / 'primary_summary.csv', [r for r in summary if r['method'] in PRIMARY])
    core.write_csv(a / 'rl3_metrics.csv', [r for r in rows if r['method'] in PRIMARY[:2]])
    core.write_csv(a / 'zero_input.csv', [r for r in rows if r['split'] == 'zero'])
    core.write_csv(a / 'per_subset_deltas.csv', paired(official, ('split', 'sample_id', 'case_id'), METRICS))
    core.write_csv(a / 'per_object_deltas.csv', paired(obj, ('split', 'sample_id'), METRICS))
    core.write_csv(a / 'object_equal_deltas.csv', paired(summary, ('split',), METRICS))
    # Show every formal case where final800 trades one structural metric for another.
    deltas = paired(official, ('split','sample_id','case_id'), METRICS)
    failures = [r for r in deltas if any(number(r, k)>0 for k in
                 ('gt_scale_aligned_nrmse','gt_axial_w1_um','background_xy_mass_fraction'))]
    core.write_csv(a / 'failure_cases.csv', failures)
    stable = []
    for (method, split, sample), items in groups(official, ('method', 'split', 'sample_id')).items():
        arrays = [np.load(root/'evaluation'/method/r['case_id']/'reconstruction.npy') for r in items]
        st = stability_values(arrays)
        st.update(method=method, split=split, sample_id=sample,
                  gt_nrmse_std=float(np.std([number(r,'gt_scale_aligned_nrmse') for r in items], ddof=1)),
                  gt_w1_std=float(np.std([number(r,'gt_axial_w1_um') for r in items], ddof=1)))
        stable.append(st)
    sm = ['shape_relative_dispersion','pairwise_axial_w1_um','mass_cv','gt_nrmse_std','gt_w1_std']
    core.write_csv(a/'stability_per_object.csv',stable)
    macro = aggregate(stable, ('method','split'),sm)
    core.write_csv(a/'stability_summary.csv',macro)
    core.write_csv(a/'stability_deltas.csv',paired(stable,('split','sample_id'),sm))
    tables={n:core.read_csv(a/f'{n}.csv') for n in ('t03_lines','t04_axial','v03_beads','point_targets','point_pairs','priority_lines','t02_tubes')}
    loc=legacy._aggregate_local(tables)
    for method in METHODS:
        rr=[r for r in tables['t02_tubes'] if r['method']==method and number(r,'threshold_fraction')==.1]
        tubes=[r for r in rr if not str(r['tube']).startswith('gap')]
        gaps=[r for r in rr if str(r['tube']).startswith('gap')]
        loc.append(dict(method=method,diagnostic='T02_continuity',items=len(tubes),
                        localized_centerline_fraction=avg(tubes,'centerline_localized_fraction'),
                        false_break_rate=np.mean([legacy._bool(r['false_break']) for r in tubes]),
                        gap_bridged_rate=np.mean([legacy._bool(r['bridged']) for r in gaps]),
                        local_depth_w1_um=avg(tubes,'local_depth_w1_um')))
    core.write_csv(a/'local_summary.csv',loc)
    repeated=[]
    for method in METHODS:
        for repeat in (1,2,3):
            p=[r for r in tables['point_targets'] if r['method']==method and int(r['repeat'])==repeat and number(r,'threshold')==.1]
            l=[r for r in tables['priority_lines'] if r['method']==method and int(r['repeat'])==repeat and number(r,'threshold')==.1]
            repeated.append(dict(method=method,repeat=repeat,point_localized_rate=np.mean([legacy._bool(r['matched']) for r in p]),
                                 local_depth_w1_um=avg(p,'local_depth_w1_um'),tail_mass=avg(p,'tail_mass_outside_pm1'),
                                 continuous_false_gap_rate=np.mean([legacy._bool(r['false_gap']) for r in l if number(r,'gap_um')==0]),
                                 gapped_bridge_rate=np.mean([legacy._bool(r['bridged']) for r in l if number(r,'gap_um')>0])))
    core.write_csv(a/'per_repeat_summary.csv', repeated)
    # Reuse all twenty real subsets per method; filter only the two historical400 comparators.
    real=Path(manifest['real_reuse']); mapping={'mean800':'mean100_final800','taylor800':'taylor100_final800',
                                               'mean_rl3':'mean_rl3','taylor_rl3_sqrt':'taylor_rl3_sqrt'}
    real_inventory=[]
    real_manifest=json.loads((real/'manifest.json').read_text())
    for old,new in [('mean800','mean100_final800'),('taylor800','taylor100_final800')]:
        assert real_manifest['checkpoints'][old]['sha256']==manifest['checkpoints'][new]['sha256']
    for field in ('45','55'):
        for subset in range(1,11):
            for method in mapping:
                d=real/field/f'subset_{subset:02d}'/method
                assert (d/'complete.json').exists() and (d/'reconstruction.npy').exists()
                real_inventory.append(dict(field=field,subset=subset,method=mapping[method],path=str(d),
                                           reconstruction_sha256=core.SHA(d/'reconstruction.npy')))
    for name in ('per_subset_metrics','field_equal_metrics','stability','axial_profiles'):
        rr=[dict(r,method=mapping[r['method']]) for r in core.read_csv(real/f'{name}.csv') if r['method'] in mapping]
        core.write_csv(a/'real_training_fields'/f'{name}.csv',rr)
    core.write_csv(a/'real_training_fields'/'reuse_inventory.csv',real_inventory)
    return rows,obj,summary,stable,macro,loc


def training(root, manifest):
    a=root/'analysis'; curves=[]; train=[]; gradients=[]; domains=[]; audit={}; schedules={}
    for method in PRIMARY[2:]:
        folder=Path(manifest['checkpoints'][method]['path']).parent
        tr=core.read_csv(folder/'training_metrics.csv'); val=core.read_csv(folder/'validation_metrics.csv')
        assert len(tr)==800 and [int(r['step']) for r in tr]==list(range(1,801))
        assert len(val)==1200 and all(len(v)==30 for v in groups(val,('step',)).values())
        train.extend(dict(r,method=method) for r in tr)
        perobj=aggregate(val,('step','sample_id'),['selection_score']+METRICS)
        cs=aggregate(perobj,('step',),['selection_score']+METRICS)
        cs.sort(key=lambda r:int(r['step']))
        curves.extend(dict(r,method=method) for r in cs)
        best=min(cs,key=lambda r:number(r,'selection_score'))
        assert int(best['step'])==(740 if method.startswith('mean') else 800)
        accepted={}
        for kind in ('gradient_diagnostics','domain_training'):
            original=[json.loads(l) for p in sorted(folder.glob(kind+'_rank*.jsonl')) for l in p.read_text().splitlines() if l.strip()]
            original=[r for r in original if r.get('phase','training')=='training']
            # Checkpoint rollback leaves repeated attempts at the same step/sample.
            # The last occurrence on the same rank is the committed replay.
            dedup={(int(r['update_step']),r['sample_id'],int(r['subset_index'])):r for r in original}
            assert len(dedup)==6400
            bystep=groups(list(dedup.values()),('update_step',))
            assert len(bystep)==800 and all(len(x)==8 for x in bystep.values())
            accepted[kind]=dict(raw_rows=len(original),committed_sample_keys=len(dedup),rollback_replays_excluded=len(original)-len(dedup))
            if kind=='gradient_diagnostics':
                for (step,),rs in bystep.items():
                    assert all(r['budget_passed'] and r['gradients_finite'] and r['shape_gradient_ratio']<=min(step/50,1)+1e-6 for r in rs)
                    gradients.append(dict(method=method,step=step,mean_ratio=avg(rs,'shape_gradient_ratio'),
                                     max_ratio=max(r['shape_gradient_ratio'] for r in rs),
                                     gradient_cosine=avg(rs,'mean_var_gradient_cosine'),tv_gradient_norm=avg(rs,'weighted_tv_q_gradient_norm')))
            else:
                schedules[method]=set(dedup)
                for (step,),rs in bystep.items():
                    assert sum(r['sample_id'].startswith('P') for r in rs)==6
                    assert sum(r['sample_id']=='real_45' for r in rs)==sum(r['sample_id']=='real_55' for r in rs)==1
                    for domain in ('simulation','real_45','real_55'):
                        selected=[r for r in rs if (r['sample_id'].startswith('P') if domain=='simulation' else r['sample_id']==domain)]
                        domains.append(dict(method=method,step=step,domain=domain,samples=len(selected),
                                            **{k:avg(selected,k) for k in ('total_loss','normalized_mean_loss','normalized_var_loss')}))
        audit[method]=dict(training_steps=800,validation_rows=1200,best_step=int(best['step']),logs=accepted)
    assert schedules[PRIMARY[2]]==schedules[PRIMARY[3]]
    core.write_csv(a/'training_curves.csv',train);core.write_csv(a/'validation_curves.csv',curves)
    core.write_csv(a/'gradient_by_step.csv',gradients);core.write_csv(a/'domain_training_curves.csv',domains)
    core.write_csv(a/'gradient_summary.csv',aggregate(gradients,('method',),['mean_ratio','max_ratio','gradient_cosine','tv_gradient_norm']))
    core.write_json(a/'training_integrity.json',dict(arms=audit,same_committed_sample_schedule=True,
                    replay_policy='last occurrence by optimizer update, sample and subset; preflight excluded'))
    return curves,train,gradients,domains


def export(fig,path,book=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    require_matplotlib_panel_alignment(fig,json_out=str(path)+'.alignment.json',strict=True)
    fig.savefig(str(path)+'.png',dpi=300)
    fig.savefig(str(path)+'.pdf')
    if path.name in ('projections','test_quality','training_validation_gradient_trends'):
        fig.savefig(str(path)+'.svg')
    env=dict(os.environ,PYTHONPATH='/tmp/speckle_figure_qa_deps')
    for tool,args in [('audit_pdf_text.py',['--min-pt','5','--json']),
                      ('audit_figure_collisions.py',['--json-out',str(path)+'.collision.json'])]:
        r=subprocess.run([sys.executable,str(SKILL/tool),str(path)+'.pdf',*args],env=env,capture_output=True,text=True)
        Path(str(path)+'.'+tool+'.log').write_text(r.stdout+r.stderr)
        if r.returncode: raise RuntimeError(f'{path} {tool}: '+r.stdout[-700:])
    if book is not None: book.savefig(fig)
    plt.close(fig)


def image(ax,im,scale,title,aspect='auto'):
    ax.imshow(np.clip(im/scale,0,1),vmin=0,vmax=1,cmap='magma',interpolation='nearest',aspect=aspect)
    ax.set_title(title,pad=9);ax.set_axis_off()


def plates(root,cases):
    figures=[]
    selected=['T02_subset_01','T03_subset_01','T04_subset_01','V03_subset_01',
              'points_z060_r01','lines_z060_r01','axial_pairs_r01']
    for cid in selected:
        case=next(c for c in cases if c['id']==cid)
        truth=core.ev._targets(case,torch.device('cpu'))['ground_truth'][0,0].numpy()
        volumes={m:np.load(root/'evaluation'/m/cid/'reconstruction.npy') for m in PRIMARY}
        volumes['GT']=truth
        scales={m:max(float(np.quantile(v,.999)),1e-30) for m,v in volumes.items()}
        # Very sparse point targets may occupy <0.1%; use max if the percentile is zero.
        scales={m:s if s>1e-29 else max(float(volumes[m].max()),1e-30) for m,s in scales.items()}
        for gid,methods in GROUPS.items():
            n=len(methods); width=4*n
            for mode in ('own','shared'):
                folder=root/'analysis/actual_reconstruction_comparison'/cid/gid/mode
                if (folder/'complete.json').exists():
                    figures.append(str((folder/'atlas.pdf').relative_to(root/'analysis')));continue
                folder.mkdir(parents=True,exist_ok=True)
                # GT retains its own whole-volume scale; both networks share one scale in B.
                shared=max(scales[m] for m in methods if m!='GT')
                chosen={m:scales[m] if mode=='own' or m=='GT' else shared for m in methods}
                core.write_json(folder/'display_scales.json',dict(mode=mode,scale=chosen,
                    whole_volume=True,percentile=99.9,zero_percentile_fallback='volume maximum',gamma=1,
                    shared_GT_scale='GT normalized independently; reconstruction methods share scale',
                    clipped_fraction={m:float((volumes[m]>chosen[m]).mean()) for m in methods},
                    xy_pitch_um=220/4/49,depth_um=list(range(10,101,10)),XZ_YZ_stretched_for_visibility=True))
                with PdfPages(folder/'atlas.pdf') as book:
                    panel_width=width*.96/(n+.10*(n-1))
                    projection_height=panel_width*(5/3)*(1+2*.38/3)/(.89-.045)
                    fig,axes=plt.subplots(3,n,figsize=(width,projection_height),gridspec_kw={'height_ratios':[3,1,1]})
                    fig.subplots_adjust(left=.025,right=.985,bottom=.045,top=.89,wspace=.10,hspace=.38)
                    for col,m in enumerate(methods):
                        v=volumes[m]
                        for row,im in enumerate((v.max(0),v.max(1),v.max(2))):
                            image(axes[row,col],im,chosen[m],LABELS[m]+' / '+['XY','XZ','YZ'][row])
                    fig.suptitle(f'{cid} | {mode} whole-volume scale | input10 / RL3',fontsize=14,y=.985)
                    export(fig,folder/'projections',book)
                    pages=1
                    if cid.startswith(('T02','T03','T04')):
                        for z in range(10):
                            fig,axes=plt.subplots(1,n,figsize=(width,panel_width/(.83-.04)))
                            fig.subplots_adjust(left=.025,right=.985,bottom=.04,top=.83,wspace=.10)
                            for col,m in enumerate(methods):image(axes[col],volumes[m][z],chosen[m],LABELS[m])
                            fig.suptitle(f'{cid} | native depth {10+10*z} um | {mode}, scale fixed over all layers',fontsize=14,y=.985)
                            export(fig,folder/f'layer_{10+10*z:03d}um',book);pages+=1
                    core.write_json(folder/'complete.json',dict(complete=True,pages=pages))
                figures.append(str((folder/'atlas.pdf').relative_to(root/'analysis')))
        print('FIGURES',cid,flush=True)
    core.write_json(root/'analysis/figure_manifest.json',figures)


def quantitative(root,obj,stable,curves,train,gradients,domains):
    a=root/'analysis';test=[r for r in obj if r['split']=='test']
    fig,axs=plt.subplots(1,3,figsize=(15,5))
    fig.subplots_adjust(left=.07,right=.97,bottom=.17,top=.77,wspace=.36)
    for ax,key,title in zip(axs,['gt_scale_aligned_nrmse','gt_axial_w1_um','background_xy_mass_fraction'],
                            ['Aligned 3D NRMSE (lower)','Axial W1 / um (lower)','Background mass (lower)']):
        for method in PRIMARY:
            rs=sorted([r for r in test if r['method']==method],key=lambda r:r['sample_id'])
            ax.errorbar(range(3),[number(r,key) for r in rs],yerr=[number(r,key+'_subset_sd') for r in rs],
                        fmt='o-',capsize=3,color=COLORS[method],label=LABELS[method])
        ax.set_xticks(range(3),['T02','T03','T04']);ax.set_title(title);ax.grid(alpha=.18)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=4,bbox_to_anchor=(.5,.985))
    export(fig,a/'test_quality')
    fig,axs=plt.subplots(1,3,figsize=(15,5));fig.subplots_adjust(left=.07,right=.97,bottom=.17,top=.77,wspace=.36)
    for ax,key,title in zip(axs,['shape_relative_dispersion','pairwise_axial_w1_um','mass_cv'],
                            ['Shape dispersion S (lower)','Pairwise axial W1 / um','Total-mass CV']):
        for m in PRIMARY:
            rs=sorted([r for r in stable if r['method']==m and r['split']=='test'],key=lambda r:r['sample_id'])
            ax.plot(range(3),[number(r,key) for r in rs],'o-',color=COLORS[m],label=LABELS[m])
        ax.set_xticks(range(3),['T02','T03','T04']);ax.set_title(title);ax.grid(alpha=.18)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=4,bbox_to_anchor=(.5,.985))
    export(fig,a/'test_stability')
    fig,axs=plt.subplots(2,3,figsize=(16,9));fig.subplots_adjust(left=.07,right=.98,bottom=.09,top=.84,wspace=.34,hspace=.40)
    specs=[(train,'total_loss','Training total loss'),(curves,'selection_score','Validation selection score'),
           (curves,'gt_scale_aligned_nrmse','Validation aligned NRMSE'),(curves,'gt_axial_w1_um','Validation axial W1 / um'),
           (gradients,'mean_ratio','Mean-structure gradient ratio'),(gradients,'gradient_cosine','Mean / variance gradient cosine')]
    for ax,(source,key,title) in zip(axs.flat,specs):
        for m in PRIMARY[2:]:
            rs=sorted([r for r in source if r['method']==m],key=lambda r:int(r['step']))
            ax.plot([int(r['step']) for r in rs],[number(r,key) for r in rs],color=COLORS[m],label=LABELS[m])
        ax.set_title(title);ax.set_xlabel('Optimizer step');ax.grid(alpha=.18)
    fig.legend(*axs.flat[0].get_legend_handles_labels(),loc='upper center',ncol=2,bbox_to_anchor=(.5,.975))
    export(fig,a/'training_validation_gradient_trends')
    # Matched-case differences preserve the paired comparison; no significance test is implied.
    deltas=core.read_csv(a/'per_subset_deltas.csv')
    fig,axs=plt.subplots(1,3,figsize=(15,5));fig.subplots_adjust(left=.07,right=.98,bottom=.17,top=.80,wspace=.34)
    for ax,k,title in zip(axs,['gt_scale_aligned_nrmse','gt_axial_w1_um','background_xy_mass_fraction'],['Delta aligned NRMSE','Delta axial W1 / um','Delta background mass']):
        for j,s in enumerate(('T02','T03','T04')):
            rs=[r for r in deltas if r['comparator']=='taylor100_final800' and r['sample_id']==s]
            ax.scatter(np.full(len(rs),j),[number(r,k) for r in rs],s=24,color=COLORS['mean100_final800'],alpha=.6)
        ax.axhline(0,color='#999999',ls='--');ax.set_xticks(range(3),['T02','T03','T04']);ax.set_title(title);ax.grid(alpha=.18)
    fig.suptitle('Mean100 minus Taylor100: matched subsets; below zero favors Mean100',y=.965,fontsize=13)
    export(fig,a/'paired_network_differences')


def profiles_and_corrections(root,cases):
    a=root/'analysis';profiles=core.read_csv(a/'fixed_profiles.csv')
    for sample in ('T02','T03','T04'):
        selected=[r for r in profiles if r['sample_id']==sample and int(r['subset'])==1 and r['method'] in PRIMARY]
        gt_t03=[]
        if sample=='T03':
            case=next(c for c in cases if c['sample']==sample and c['subset']==1)
            truth=core.ev._targets(case,torch.device('cpu'))['ground_truth'][0,0].numpy()
            _,gt_t03=core.local.t03(truth,truth,{'method':'GT'})
        regions=list(dict.fromkeys(r['region'] for r in selected))
        count=len(regions);nr=(count+3)//4
        fig,axs=plt.subplots(nr,4,figsize=(16,3.4*nr),squeeze=False)
        fig.subplots_adjust(left=.05,right=.98,bottom=.06,top=.90,wspace=.30,hspace=.48)
        for index,ax in enumerate(axs.flat):
            if index>=count:ax.set_axis_off();continue
            region=regions[index]
            for method in PRIMARY:
                rs=sorted([r for r in selected if r['region']==region and r['method']==method],key=lambda r:number(r,'coordinate'))
                xx=[number(r,'coordinate') for r in rs];y=np.array([number(r,'value') for r in rs]);y/=max(y.max(),1e-30)
                ax.plot(xx,y,'.-',color=COLORS[method],label=LABELS[method],ms=3)
            if rs and rs[0].get('gt_value'):
                y=np.array([number(r,'gt_value') for r in rs]);y/=max(y.max(),1e-30)
                ax.plot(xx,y,'k--',label='GT')
            elif gt_t03:
                gs=sorted([r for r in gt_t03 if str(r['region'])==str(region)],key=lambda r:r['coordinate'])
                y=np.array([r['value'] for r in gs]);y/=max(y.max(),1e-30)
                ax.plot([r['coordinate'] for r in gs],y,'k--',label='GT')
            ax.set_title(f'{sample} region {region}');ax.set_xlabel('Transverse pixel' if sample=='T03' else 'Native depth / um');ax.set_ylim(-.03,1.05);ax.grid(alpha=.15)
        fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper center',ncol=5,bbox_to_anchor=(.5,.985),fontsize=10)
        export(fig,a/f'{sample}_final_fixed_profiles')
        cid=sample+'_subset_01'
        for method in PRIMARY[2:]:
            base=root/'evaluation'/method/cid
            pred=np.load(base/'reconstruction.npy');anchor=np.load(base/'physical_anchor.npy');delta=np.load(base/'effective_correction.npy')
            scale=max(float(np.quantile(pred,.999)),1e-30);dscale=max(float(np.quantile(abs(delta),.999)),1e-30)
            ascale=max(float(np.quantile(anchor,.999)),1e-30)
            fig,axs=plt.subplots(1,3,figsize=(12,5));fig.subplots_adjust(left=.025,right=.985,bottom=.08,top=.80,wspace=.1)
            image(axs[0],anchor.max(0),ascale,'Calibrated anchor / own scale')
            image(axs[1],pred.max(0),scale,'Final reconstruction')
            # Signed correction shown in the fixed native 50um plane, not absolute MIP.
            axs[2].imshow(delta[4],vmin=-dscale,vmax=dscale,cmap='RdBu_r',aspect='auto');axs[2].set_title('Signed correction / native 50 um');axs[2].set_axis_off()
            fig.suptitle(f'{cid} | {LABELS[method]} | correction red positive / blue negative',fontsize=12,y=.965)
            path=a/'anchor_and_correction'/f'{sample}_{method}'
            export(fig,path)
            core.write_json(Path(str(path)+'.display.json'),dict(anchor_scale=ascale,output_scale=scale,correction_abs_scale=dscale,correction_layer_um=50,
                            correction_clipped_fraction=float((abs(delta)>dscale).mean()),no_per_layer_normalization=True))
        # All ten subsets, fixed common whole-volume scale per network, one XY plate.
        for method in PRIMARY[2:]:
            vs=[np.load(root/'evaluation'/method/f'{sample}_subset_{k:02d}'/'reconstruction.npy') for k in range(1,11)]
            scale=max(float(np.quantile(v,.999)) for v in vs)
            panel_width=18*.96/(5+.1*4)
            fig,axs=plt.subplots(2,5,figsize=(18,panel_width*2.2/.86));fig.subplots_adjust(left=.025,right=.985,bottom=.04,top=.90,wspace=.1,hspace=.2)
            for k,ax in enumerate(axs.flat):image(ax,vs[k].max(0),scale,f'Subset {k+1:02d}')
            fig.suptitle(f'{sample} | {LABELS[method]} | identical whole-volume scale across ten subsets',fontsize=14,y=.98)
            export(fig,a/'subset_variability'/f'{sample}_{method}')


def markdown_table(rows,keys,headers=None):
    lines=['|'+'|'.join(headers or keys)+'|','|'+'|'.join('---' for _ in keys)+'|']
    for r in rows:
        values=[]
        for k in keys:
            v=r.get(k,'');v=LABELS.get(v,v) if isinstance(v,str) else v
            if isinstance(v,str) and k not in ('method','sample_id','field_id','diagnostic'):
                try:v=float(v)
                except ValueError:pass
            if isinstance(v,(float,np.floating)):v=f'{v:.5f}' if math.isfinite(v) else 'NA'
            values.append(str(v))
        lines.append('|'+'|'.join(values)+'|')
    return '\n'.join(lines)


def report(root,manifest,summary,obj,stable,macro,loc):
    a=root/'analysis'
    ss=sorted([r for r in summary if r['split']=='test' and r['method'] in PRIMARY],key=lambda r:PRIMARY.index(r['method']))
    lookup={r['method']:r for r in ss};m=lookup['mean100_final800'];t=lookup['taylor100_final800']
    keys=['gt_scale_aligned_nrmse','gt_axial_w1_um','background_xy_mass_fraction']
    conclusion='；'.join(f'{title}{"降低" if number(m,k)<number(t,k) else "升高"}{abs((number(m,k)/number(t,k)-1)*100):.1f}%' for k,title in zip(keys,['尺度对齐NRMSE','轴向W1','背景质量']))
    text=['# 当前Mean100基线：四方法整体分析','',
          '## 结论与适用范围','',f'正式仿真测试集对象等权平均，相对Taylor100 final800：{conclusion}。',
          '当前基线仍按用户选择保留Mean100 final800。各指标、局部分辨和稳定性必须分别看；不把某个平均指标改善等同于全部结构更清楚。',
          '直白地说：Mean100在T02/T03的整体形状误差更低，三个对象的整体深度W1都更低；但T02背景更多、跨子集波动更大，T04形状误差还比Taylor100更高。T03三线分开率同为66.7%，T04的20–40 μm双层分开率两者均为0%。所以本轮是混合收益，不能宣称Mean基础已全面提高分辨率或稳定性。',
          '两网络均为同一单种子、混合数据训练的800步权重。Mean100 best740仅为补充；Taylor best800模型参数与final800逐项相同。100是mean结构梯度预算≤100%，不是100帧。',
          '主比较：Mean-RL3 / Taylor-RL3-sqrt / Taylor100 final800 / Mean100 final800；均为10帧RL3输入。没有把旧Taylor400混入Taylor100。',
          '', '## 正式测试总体质量','',
          markdown_table(ss,['method','gt_raw_nrmse','gt_scale_aligned_nrmse','gt_axial_w1_um','gt_xy_mip_ssim','background_xy_mass_fraction'],
                            ['方法','原始NRMSE↓','对齐NRMSE↓','轴向W1 μm↓','XY SSIM↑','背景质量↓']),
          '', '先对每对象10子集取平均，再对T02–T04三个对象等权平均；质量图误差线为10个技术子集的样本SD，不是三个独立训练种子的置信区间。单种子，不作显著性检验。RL原始强度与网络亮度校准不同，原始误差不宜单独排名。',
          '', '## 逐对象与跨子集稳定性','',
          markdown_table(sorted([r for r in obj if r['split']=='test' and r['method'] in PRIMARY],key=lambda r:(r['sample_id'],PRIMARY.index(r['method']))),
                         ['sample_id','method',*keys],['对象','方法','对齐NRMSE↓','轴向W1 μm↓','背景质量↓']),'',
          markdown_table([r for r in macro if r['split']=='test' and r['method'] in PRIMARY],['method','shape_relative_dispersion','pairwise_axial_w1_um','mass_cv'],
                         ['方法','形状波动S↓','子集两两轴向W1 μm↓','总质量CV↓']),
          '', 'S是各体总质量归一化后的相对离散度；对子集45对计算轴向W1。稳定不等于正确：始终模糊、恒定深度的结果也可能很稳定。误差波动另见stability_per_object.csv。',
          '', '## 局部分辨、断裂、轴向诊断','',
          markdown_table([r for r in loc if r['method'] in PRIMARY and r['diagnostic']=='T03_three_lines'],['method','localized_rate','separated_rate','false_break_rate'],
                         ['方法','T03三线定位率↑','三线分开率↑','存在线条断裂的组比例↓']), '',
          markdown_table([r for r in loc if r['method'] in PRIMARY and r['diagnostic']=='T04_axial_pairs'],['method','localized_rate','separated_rate_20_40um','expected_layer_energy_fraction'],
                         ['方法','T04所有预期层定位率↑','20–40 μm双层分开率↑','预期层能量比例↑']), '',
          markdown_table([r for r in loc if r['method'] in PRIMARY and r['diagnostic']=='T02_continuity'],['method','localized_centerline_fraction','false_break_rate','gap_bridged_rate','local_depth_w1_um'],
                         ['方法','T02中心线覆盖率↑','管段断裂率↓','已知间隙错连率↓','局部深度W1 μm↓']),
          '', '主阈值为体最大值10%，同时保存5%与20%敏感性结果。T04的10 μm间距只评价原生层定位与能量，双峰分开率记不适用；不以插值制造分辨结论。T02断裂恢复与错误跨间隙连接分开记录。',
          'V03的60念珠、历史点目标/线条/轴向对共33例、三次照明重复与零输入另列，不与正式测试样本混合平均。局部量化先通过GT正对照。',
          '特别注意：整体轴向W1只看全部体素合并后的深度质量分布，即使全局深度对了，同一位置上的双层仍可能合成一团。T04目前正是这个风险；其对齐NRMSE约0.956（Mean100）与0.942（Taylor100），没有恢复出可靠局部双层。',
          'T04两网络的对齐NRMSE也都高于Mean-RL3的0.895，不能据平均指标就称网络在每个对象上都超过RL。T04“所有预期层定位率”同时含单层控制区，并不表示25%的双层已经分开。',
          'T02的连续管段断裂率从Taylor100的10.0%降到Mean100的3.1%，但错误连接已知间隙的比例从10.0%升到16.7%，即“更连贯”也伴随更容易错误连接。',
          '', '## 清晰图册入口','',
          '按用户之前指定的拆分方式：A组为GT＋Mean-RL3＋Taylor-RL3-sqrt＋Mean100；B组为GT＋Taylor100＋Mean100。每组分别提供整体归一化和重建共同尺度。',
          '所有层共用该重建体的99.9百分位尺度（极稀疏点的零百分位回退到最大值），不逐层归一化；记录截断比例。GT单独整体归一化以供结构参照，不用于亮度比较。XY坐标为像素，横向采样220/4/49 μm；XZ/YZ仅显示拉伸，量化使用原生10 μm层间距。', '']
    for s in ('T02','T03','T04','V03'):
        text += [f'- {s}：'+ ' · '.join(f'[{label}]({"actual_reconstruction_comparison/"+s+"_subset_01/"+g+"/"+mode+"/atlas.pdf"})' for label,g,mode in [('RL与Mean100','A_RL_vs_Mean100','own'),('两网络','B_Taylor100_vs_Mean100','own'),('两网络共同尺度','B_Taylor100_vs_Mean100','shared')])]
    text += ['', '[质量图](test_quality.png) · [稳定性图](test_stability.png) · [配对子集差值](paired_network_differences.png) · [训练/验证/梯度趋势](training_validation_gradient_trends.png)',
             '', '每个T02–T04图册包含投影和全部十层；局部剖面、基础体/有效修正和十子集波动分别在相应子目录。历史诊断图保留60 μm代表例及轴向双层代表例。',
             '', '## best补充与训练完整性','',
             markdown_table([r for r in summary if r['split']=='test' and r['method'] not in PRIMARY[:2]],['method',*keys],
                            ['模型/权重','对齐NRMSE↓','轴向W1 μm↓','背景质量↓']),
             '', '两组各800条训练、1200条周期验证记录，best仅以仿真验证物理分数选择。完整40个验证节点在validation_curves.csv。',
             'Taylor训练恢复期间621–625步有38条重复尝试日志；梯度/分域曲线按同一步、样本、子集最后一次记录去重，保留6400个已提交样本键，不把重放当额外更新。Mean无重复。两组提交样本顺序键一致、每步6仿真＋45和55各1；详见training_integrity.json。',
             '', '## 菠菜根：两训练视场的完整复用诊断','',
             '45与55各10子集，共四方法80例。复用已核验的全幅1029×1421输出并校验权重/文件指纹；无需再次消耗GPU。它们均参与训练且无GT，90帧是训练约束，不是独立测试。不能据此报告真实NRMSE、真实深度准确率或泛化增益。','',
             markdown_table([{**r,**{k:number(r,k) for k in ('constraint90_mean_aligned_nrmse','constraint90_variance_aligned_nrmse','axial_centroid_um')}}
                             for r in core.read_csv(a/'real_training_fields/field_equal_metrics.csv') if r['field_id']!='field_macro'],
                            ['field_id','method','constraint90_mean_aligned_nrmse','constraint90_variance_aligned_nrmse','axial_centroid_um'],
                            ['训练视场','方法','90帧均值投影对齐NRMSE↓','90帧方差投影对齐NRMSE↓','重建轴向质心 μm']),'',
             '这是传感器统计投影误差，不是真实体数据误差。Mean100两视场方差统计对齐误差更低，但均值一致性没有一致胜出；其质心比Taylor100更深，不能在无GT时判定哪个深度正确。',
             'Mean100有效修正/校准基础的L2比值在仿真T02/T03/T04约为40.9/21.3/11.0，在真实45/55约为52.0/56.1。网络已经大幅重塑基础，不是仅作很小修补；基础稳定性不会自动转化成输出稳定性。',
             f'[菠菜根分组清晰图册]({manifest["real_grouped_figures"]}) · [四方法逐视场统计](real_training_fields/field_equal_metrics.csv) · [20子集明细](real_training_fields/per_subset_metrics.csv) · [真实稳定性](real_training_fields/stability.csv)',
             '', '## 文件与复现','',
             'metrics.csv为五个评价角色×94例=470行，主比较为前四方法376例；Mean best740增加94例补充。输入两路RL结果原样复用；新网络预测282例。evaluation中保存浮点NPY/TIFF，网络还保存基础体、校准前重建和有效修正。',
             'per_subset_deltas/per_object_deltas/object_equal_deltas均为Mean100 final800减去对比方法；NRMSE、W1和背景负值为好，SSIM正值为好。',
             '清晰图使用nature-figure的整体现示尺度、面板对齐、字体与碰撞审计；PDF为可编辑文字，PNG为300 dpi屏幕检查版，不宣称已满足某期刊最终排版。',
             '新脚本与权重、数据清单、当前基线登记均有快照；不修改训练语义、不自动更换基线、不删除历史结果。']
    (a/'REPORT_ZH.md').write_text('\n'.join(text),encoding='utf-8')
    (root/'README_ZH.md').write_text('# Mean100整体分析\n\n[完整报告](analysis/REPORT_ZH.md)\n',encoding='utf-8')


def main(root):
    manifest,cases=core.setup(root)
    assert (root/'local_complete.json').exists()
    contract=dict(question='Does Mean100 improve structural quality and stability, and at what resolution/background trade-off?',
                  conclusion='To be determined from paired metrics and native-layer images, not assumed from baseline promotion.',
                  evidence_roles=['object-level quality','matched subset differences','stability','local resolution failure boundaries','training provenance'],
                  archetype='image plate + quant',backend='Python',reuse='structural adaptation of existing V3 analysis definitions',
                  primary=list(PRIMARY),supplementary=['mean100_best740'],unit='object; ten technical subsets; one training seed',
                  uncertainty='individual object means and all matched subsets shown; no invented CI or significance test',
                  transforms='whole-volume linear p999; no layer normalization; sparse zero-p999 fallback max; local profiles unit peak only for display',
                  export='wide screen 300dpi PNG, editable PDF; key SVG; minimum 5pt; not journal submission layout')
    contract['source_warning_review']={
        'PNG_not_TIFF':'PNG is a screen preview; floating-point reconstruction TIFF and editable PDF are delivered separately.',
        '300_not_600dpi':'Screen comparison requested, not a submission raster specification.',
        'wide_screen_size':'Wide figures intentionally prioritize clear layer inspection; no journal sizing claim.',
        'uncertainty':'Quality mean±technical-subset SD; matched scatter shows all ten subsets; stability and training curves are deterministic summaries, not independent-seed inference.'}
    core.write_json(root/'analysis/figure_contract.json',contract)
    rows,obj,summary,stable,macro,loc=summarize(root,manifest,cases)
    curves,train,gradients,domains=training(root,manifest)
    quantitative(root,obj,stable,curves,train,gradients,domains)
    profiles_and_corrections(root,cases)
    plates(root,cases)
    report(root,manifest,summary,obj,stable,macro,loc)
    shutil.copy2(Path(__file__),root/'report_source_snapshot.py')
    core.write_json(root/'complete.json',dict(complete=True,primary_methods=PRIMARY,primary_cases=376,
                    supplementary_cases=94,real_reused_cases=80,network_predictions=282,
                    report='analysis/REPORT_ZH.md',source_sha256=core.SHA(Path(__file__))))
    print('ANALYSIS COMPLETE',root,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);args=p.parse_args();main(args.output)
