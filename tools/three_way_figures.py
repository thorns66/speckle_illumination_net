"""Fixed-scale review figures, per-repeat summaries, and explicit failures."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.three_way_experiment import OUTPUT,PRIORITY
from tools.priority_validation_analysis import _read_yxz


def make(tables,summary):
    from tools.three_way_report import csv_write,csv_read
    analysis=OUTPUT/'analysis';figures=analysis/'cases';figures.mkdir(parents=True,exist_ok=True)
    methods=[r['method'] for r in summary]
    repeat_rows=[]
    for method in methods:
        for repeat in [1,2,3]:
            p=[r for r in tables['point_targets'] if r['method']==method and int(r['repeat'])==repeat and float(r['threshold'])==.1]
            pairs=[r for r in tables['point_pairs'] if r['method']==method and int(r['repeat'])==repeat and float(r['threshold'])==.1]
            lines=[r for r in tables['lines'] if r['method']==method and int(r['repeat'])==repeat and float(r['threshold'])==.1]
            mean=lambda rows,k:float(np.mean([float(x[k]) for x in rows]))
            rate=lambda rows,k:float(np.mean([x[k] in ['True','true','1'] for x in rows]))
            repeat_rows.append({'method':method,'repeat':repeat,'single_localization_rate':rate([x for x in p if x['kind']=='single'],'matched'),
                'local_depth_w1_um':mean(p,'local_depth_w1_um'),
                'lateral_pair_separation_rate':rate([x for x in pairs if x['scene_id'].startswith('points')],'separated'),
                'axial_pair_separation_rate':rate([x for x in pairs if x['scene_id']=='axial_pairs'],'separated'),
                'false_gap_rate':rate([x for x in lines if float(x['gap_um'])==0],'false_gap'),
                'bridge_rate':rate([x for x in lines if float(x['gap_um'])>0],'bridged')})
    csv_write(analysis/'per_repeat_summary.csv',repeat_rows)
    rows=[r for r in tables['metrics'] if r['split']=='priority']
    lookup={(r['method'],r['case_id']):r for r in rows}
    failures=[]
    for method in methods:
        if method.startswith('baseline'):continue
        baseline='baseline_'+method.split('_')[-1]
        for row in [r for r in rows if r['method']==method]:
            ref=lookup[baseline,row['case_id']]
            failures.append({'method':method,'baseline':baseline,'case_id':row['case_id'],
               'shape_error_increase':float(row['gt_scale_aligned_nrmse'])-float(ref['gt_scale_aligned_nrmse']),
               'depth_error_increase_um':float(row['gt_axial_w1_um'])-float(ref['gt_axial_w1_um']),
               'shape_error':float(row['gt_scale_aligned_nrmse']),'baseline_shape_error':float(ref['gt_scale_aligned_nrmse'])})
    failures.sort(key=lambda r:r['shape_error_increase'],reverse=True)
    csv_write(analysis/'failure_cases.csv',failures)
    chosen=list(dict.fromkeys(['points_z060_r01','lines_z060_r01','axial_pairs_r01']+
                             [r['case_id'] for r in failures[:6]]))
    sigma=float(np.load(OUTPUT/'illumination/calibration_moments.npz')['sigma'])
    for case in chosen:
        truth=_read_yxz(PRIORITY/'generated'/case/'prepared.mat','ground_truth')*sigma
        volumes=[truth]
        for method in methods:
            experiment,role=method.split('_')
            volumes.append(np.load(OUTPUT/'evaluation'/experiment/role/case/'reconstruction.npy'))
        vmax=max(float(x.max()) for x in volumes)
        flat=int(np.argmax(truth.sum(0)));y,x=np.unravel_index(flat,truth.shape[1:])
        profile=lambda a:a[:,max(0,y-2):y+3,max(0,x-2):x+3].sum((1,2))
        gt_profile=profile(truth);gt_profile=gt_profile/max(float(gt_profile.sum()),1e-30)
        fig,axes=plt.subplots(3,len(volumes),figsize=(25,8))
        for col,(name,volume) in enumerate(zip(['GT in fixed units',*methods],volumes)):
            axes[0,col].imshow(volume.max(0),cmap='magma',vmin=0,vmax=vmax)
            axes[0,col].set_title(name);axes[0,col].axis('off')
            axes[1,col].imshow(volume.max(1),cmap='magma',vmin=0,vmax=vmax,aspect='auto',extent=[0,292,105,5])
            axes[1,col].set_xlabel('X, um');axes[1,col].set_ylabel('Z, um')
            p=profile(volume);p=p/max(float(p.sum()),1e-30)
            axes[2,col].plot(np.arange(10,101,10),gt_profile,'k--',label='GT')
            axes[2,col].plot(np.arange(10,101,10),p,label=name);axes[2,col].set_ylim(0,1)
            axes[2,col].set_xlabel('Z, um');axes[2,col].set_ylabel('Local mass fraction')
        fig.suptitle(case+f' | one shared intensity scale; fixed profile XY=({x},{y})')
        fig.tight_layout();fig.savefig(figures/(case+'.png'),dpi=150);plt.close(fig)
        # A second view isolates structure: one normalization per entire 3-D
        # volume, identical for every layer of that volume; raw files stay unchanged.
        fig,axes=plt.subplots(2,len(volumes),figsize=(25,6))
        for col,(name,volume) in enumerate(zip(['GT',*methods],volumes)):
            shown=volume/max(float(volume.max()),1e-30)
            axes[0,col].imshow(shown.max(0),cmap='magma',vmin=0,vmax=1)
            axes[0,col].set_title(name);axes[0,col].axis('off')
            axes[1,col].imshow(shown.max(1),cmap='magma',vmin=0,vmax=1,aspect='auto',extent=[0,292,105,5])
            axes[1,col].set_xlabel('X, um');axes[1,col].set_ylabel('Z, um')
        fig.suptitle(case+' | structure display: one scale per whole volume, never per layer')
        fig.tight_layout();fig.savefig(figures/(case+'_shape.png'),dpi=150);plt.close(fig)
    gains=csv_read(analysis/'fixed_gt_gain_audit.csv')
    fig,axes=plt.subplots(3,2,figsize=(11,11))
    for i,sample in enumerate(['P08','P09','P10']):
        for j,keys in enumerate([['old_variance_loss','corrected_variance_loss'],['old_mean_loss','calibrated_mean_loss']]):
            ax=axes[i,j]
            for key in keys:
                for repeat in [1,2,3]:
                    selected=[r for r in gains if r['sample_id']==sample and int(r['repeat'])==repeat]
                    selected.sort(key=lambda r:float(r['gain']))
                    ax.plot([float(r['gain']) for r in selected],[float(r[key]) for r in selected],
                            marker='o',alpha=.65,label=f'{key}, repeat {repeat}')
            ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_title(sample);ax.set_xlabel('Fixed GT gain')
            ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(analysis/'fixed_gt_gain_audit.png',dpi=160);plt.close(fig)
    # Loss definitions differ: plot each objective separately, not as one ranking.
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,experiment in zip(axes,['e1','e2','e3']):
        rows=csv_read(OUTPUT/experiment/'validation_metrics.csv')
        steps=sorted({int(r['step']) for r in rows})
        ax.plot(steps,[np.mean([float(r['selection_score']) for r in rows if int(r['step'])==step]) for step in steps])
        best=next(r['weight_step'] for r in summary if r['method']==experiment+'_best')
        ax.axvline(best,color='red',linestyle='--',label=f'best step {best}')
        ax.set_title(experiment+' native validation objective');ax.set_xlabel('Optimizer step');ax.legend()
    fig.tight_layout();fig.savefig(analysis/'checkpoint_selection.png',dpi=160);plt.close(fig)
    return {'case_figures':chosen,'largest_shape_regressions':failures[:6],'repeat_rows':len(repeat_rows)}
