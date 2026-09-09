"""CPU-only paired challenge report using fixed V3 structural rules."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets.matlab_multivolume_dataset import load_inference_input
from tools.v3_compare_evaluation import _structure_row, csv_write
from tools.v3_compare_local_audit import t03, t04
from tools.run_correlation_challenge import DATA, TRAIN, SAMPLES, ARMS, save, log

METHODS = ('mean_rl3', 'taylor_rl3', *ARMS)
METRICS = ('gt_raw_nrmse', 'gt_scale_aligned_nrmse', 'gt_axial_w1_um', 'gt_axial_mass_l1',
           'gt_support_outside_pm10_mass', 'gt_xy_mip_ssim', 'background_xy_mass_fraction')


def average(rows, keys):
    return {k:float(np.mean([float(r[k]) for r in rows])) for k in keys}


def volumes(output, sample, group, role):
    directory = Path(sample['challenge_dir']) if group in ('low','high') else DATA/sample['sample_id']
    index = (1 if group=='low' else 2) if group in ('low','high') else int(group.split('_')[1])
    raw = load_inference_input(directory,index)
    result = {'mean_rl3':raw['g_mean'][0], 'taylor_rl3':raw['f_var'][0]}
    for arm in ARMS:
        path = (output/'predictions'/arm/role/sample['sample_id']/group/'reconstruction.npy'
                if group in ('low','high') else
                TRAIN/'evaluation'/arm/role/f'{sample["sample_id"]}_subset_{index:02d}'/'reconstruction.npy')
        result[arm] = np.load(path)
    return result


def plot_comparison(output, name, role, truth, values):
    directory = output/'figures'/name/role
    directory.mkdir(parents=True,exist_ok=True)
    for display in ('shape','shared'):
        shared = max(float(v.max()) for group in values.values() for v in group.values())
        for projection, axis in [('xy',0),('xz',1),('yz',2)]:
            fig, axes = plt.subplots(2,6,figsize=(18,7),constrained_layout=True)
            for row, group in enumerate(('low','high')):
                for col, (method, volume) in enumerate([('GT',truth),*values[group].items()]):
                    # GT has a separate native-density scale in shared plots;
                    # all reconstructions share ONE limit across both subsets.
                    scale = max(float(volume.max()),1e-30) if display=='shape' or method=='GT' else max(shared,1e-30)
                    axes[row,col].imshow((volume/scale).max(axis),cmap='magma',vmin=0,vmax=1,
                                         aspect='auto' if axis else 'equal',interpolation='nearest')
                    axes[row,col].set_title(f'{group} / {method}',fontsize=10)
                    axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
            fig.suptitle(f'{name} {role} {projection.upper()} | {display}; full-volume scales; GT native scale')
            fig.savefig(directory/f'{display}_{projection}.png',dpi=150); plt.close(fig)
        # All native layers, fixed in advance; no per-slice normalization.
        fig, axes = plt.subplots(20,6,figsize=(15,45),constrained_layout=True)
        for gi,group in enumerate(('low','high')):
            for col,(method,volume) in enumerate([('GT',truth),*values[group].items()]):
                scale = max(float(volume.max()),1e-30) if display=='shape' or method=='GT' else max(shared,1e-30)
                for zi in range(10):
                    ax=axes[gi*10+zi,col]
                    ax.imshow(volume[zi]/scale,cmap='magma',vmin=0,vmax=1,interpolation='nearest')
                    ax.set_xticks([]);ax.set_yticks([])
                    if zi==0:ax.set_title(f'{group}/{method}')
                    if col==0:ax.set_ylabel(f'{10*(zi+1)} um')
        fig.savefig(directory/f'{display}_native_layers.png',dpi=100);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,group in zip(axs,('low','high')):
        for method,v in [('GT',truth),*values[group].items()]:
            profile=v.sum((1,2),dtype=np.float64)
            ax.plot(np.arange(10,101,10),profile/max(profile.sum(),1e-30),'.-',label=method)
        ax.set_title(group);ax.set_xlabel('Native depth (um)');ax.set_ylabel('Fraction of volume mass');ax.legend(fontsize=8)
    fig.savefig(directory/'depth_profiles.png',dpi=150);plt.close(fig)


def report(output,run):
    manifest=json.loads(Path(run['manifest']).read_text())
    rows=[];local=[];profiles=[];selection=[]
    control={(r['sample_id'],r['group'],r['method']):r for r in json.loads((output/'control_physics.json').read_text())}
    for sample in manifest['samples']:
        name=sample['sample_id']
        with h5py.File(DATA/name/'prepared.mat') as h:
            truth=np.asarray(h['ground_truth'],dtype=np.float32).transpose(0,2,1)
        for group in ('low','high'):
            selection.append({'sample_id':name,'group':group,'score':sample[group]['score'],
                              'input_indices':sample[group]['input_indices'],**sample[group]['brightness']})
        for role in ('final','best'):
            display={}
            for group in ['low','high',*[f'random_{i:02d}' for i in range(1,11)]]:
                data=volumes(output,sample,group,role)
                if group in ('low','high'):display[group]=data
                for method,pred in data.items():
                    meta={'sample_id':name,'group':group,'method':method,'checkpoint_role':role}
                    scores=_structure_row(pred,truth)
                    if group in ('low','high'):
                        if method in ARMS:
                            record=json.loads((output/'predictions'/method/role/name/group/'complete.json').read_text())
                            scores.update(record['common_scores'])
                        else:
                            scores.update({k:v for k,v in control[name,group,method].items() if k.startswith('common_')})
                    rows.append({**meta,**scores})
                    if name in ('T03','T04'):
                        lr,pr=(t03 if name=='T03' else t04)(pred,truth,meta)
                        local.extend(lr);profiles.extend(pr)
            plot_comparison(output,name,role,truth,display)
            log(f'Reported {name}/{role}')
        fig,axs=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
        corr=np.load(Path(sample['challenge_dir'])/'correlation_signed.npy')
        axs[0].imshow(corr,vmin=-1,vmax=1,cmap='coolwarm');axs[0].set_title(f'{name}: 100-frame signed Pearson')
        random=np.load(Path(sample['challenge_dir'])/'random_candidate_scores.npy')
        axs[1].hist(random,bins=35,color='gray')
        for group,color in [('low','blue'),('high','red')]:
            axs[1].axvline(sample[group]['score'],color=color,label=group)
        axs[1].legend();axs[1].set_title('1,000 random subsets vs selected')
        axs[2].plot([r['score'] for r in sample['original_random_subsets']],'o-')
        axs[2].axhline(sample['low']['score'],color='blue');axs[2].axhline(sample['high']['score'],color='red')
        axs[2].set_title('Original 10 random test subsets')
        fig.savefig(output/'figures'/f'{name}_correlation.png',dpi=150);plt.close(fig)
    # Collapse subsets within object before object-equal aggregation.
    per_object=[]
    for name in SAMPLES:
        for role in ('final','best'):
            for method in METHODS:
                for group in ('low','high','random'):
                    subset=[r for r in rows if r['sample_id']==name and r['checkpoint_role']==role
                            and r['method']==method and (r['group'].startswith('random_') if group=='random' else r['group']==group)]
                    item={'sample_id':name,'checkpoint_role':role,'method':method,'group':group,**average(subset,METRICS)}
                    for k in METRICS:
                        item[k+'_min']=min(r[k] for r in subset);item[k+'_max']=max(r[k] for r in subset)
                    per_object.append(item)
    macro=[];differences=[];method_differences=[]
    for role in ('final','best'):
        for method in METHODS:
            for group in ('low','high','random'):
                subset=[r for r in per_object if r['checkpoint_role']==role and r['method']==method and r['group']==group]
                macro.append({'checkpoint_role':role,'method':method,'group':group,**average(subset,METRICS)})
            for name in (*SAMPLES,'object_macro'):
                source=macro if name=='object_macro' else [r for r in per_object if r['sample_id']==name]
                selected={r['group']:r for r in source if r['checkpoint_role']==role and r['method']==method}
                differences.append({'sample_id':name,'checkpoint_role':role,'method':method,'difference':'low_minus_high',
                                    **{k:selected['low'][k]-selected['high'][k] for k in METRICS}})
        for name in (*SAMPLES,'object_macro'):
            source=macro if name=='object_macro' else [r for r in per_object if r['sample_id']==name]
            for group in ('low','high','random'):
                selected={r['method']:r for r in source if r['checkpoint_role']==role and r['group']==group}
                for method in ARMS:
                    for reference in ('mean_rl3','taylor_rl3','baseline'):
                        if method==reference:continue
                        method_differences.append({'sample_id':name,'checkpoint_role':role,'group':group,'method':method,
                                                   'reference':reference,**{k:selected[method][k]-selected[reference][k] for k in METRICS}})
    for filename,table in [('metrics',rows),('per_object',per_object),('object_macro',macro),('low_minus_high',differences),
                           ('method_differences',method_differences),('local_metrics',local),('local_profiles',profiles),('selection',selection)]:
        csv_write(output/f'{filename}.csv',table)
    # Local curves share a scale across methods AND low/high for each region.
    for name in ('T03','T04'):
        for role in ('final','best'):
            selected=[r for r in profiles if r['sample_id']==name and r['checkpoint_role']==role and r['group'] in ('low','high')]
            regions=sorted(set(r['region'] for r in selected),key=str)
            fig,axs=plt.subplots(len(regions),2,figsize=(12,3*len(regions)),squeeze=False,constrained_layout=True)
            for i,region in enumerate(regions):
                region_rows=[r for r in selected if r['region']==region]
                scale=max(max(r['value'] for r in region_rows),1e-30)
                for j,group in enumerate(('low','high')):
                    ax=axs[i,j]
                    for method in METHODS:
                        rr=sorted([r for r in region_rows if r['group']==group and r['method']==method],key=lambda r:r['coordinate'])
                        ax.plot([r['coordinate'] for r in rr],[r['value']/scale for r in rr],'.-',label=method)
                    ax.set_title(f'{name}/{region}/{group}');ax.legend(fontsize=7);ax.grid(alpha=.2)
            fig.savefig(output/'figures'/name/role/'local_profiles.png',dpi=120);plt.close(fig)
    lines=['# 低／高相关 10 帧测试对比','',
           '主比较使用第 400 步 final；best 仅作补充。没有重新训练或用这些测试结果选模。',
           '相关性是相机浮点图全视野的空间 Pearson，子集分数为 45 对绝对值均值；搜索极值不是已证明的全局极值。',
           '这是从已采集的 100 帧中选 10 帧，不等同于仅采集 10 帧。低相关不保证恢复更好。','',
           '## 子集相关性','', '| 对象 | 低相关 | 原十组均值 | 高相关 | 两组重叠帧数 |','|---|---:|---:|---:|---:|']
    for s in manifest['samples']:
        lines.append(f"| {s['sample_id']} | {s['low']['score']:.5f} | {np.mean([r['score'] for r in s['original_random_subsets']]):.5f} | {s['high']['score']:.5f} | {s['overlap_count']} |")
    lines += ['', '## 第 400 步：三个对象等权平均','', '| 方法 | 子集 | 对齐 NRMSE ↓ | 轴向 W1 μm ↓ | XY SSIM ↑ |','|---|---|---:|---:|---:|']
    for r in macro:
        if r['checkpoint_role']=='final':
            lines.append(f"| {r['method']} | {r['group']} | {r['gt_scale_aligned_nrmse']:.4f} | {r['gt_axial_w1_um']:.2f} | {r['gt_xy_mip_ssim']:.4f} |")
    lines += ['', '## 低相关相对高相关的实际变化','']
    for r in differences:
        if r['sample_id']=='object_macro' and r['checkpoint_role']=='final':
            lines.append(f"- {r['method']}：对齐 NRMSE 差 {r['gt_scale_aligned_nrmse']:+.4f}，W1 差 {r['gt_axial_w1_um']:+.2f} μm（均为低减高，负数改善）。")
    lines += ['', '## 图像与解释限制','',
              'figures 中提供低／高相关并排的 XY/XZ/YZ、全部十个原生单层和深度曲线；shape 按整个三维体归一化，shared 在同对象两组选帧及所有重建方法之间共用尺度，GT 使用独立原生密度尺度。没有逐层归一化。',
              'T03/T04 局部指标与曲线在 local_metrics.csv、local_profiles.csv 和相应图中。T04 的 10 μm 相邻层没有原生谷值，不算作可证明的双峰分开。',
              '总亮度、帧间亮度变化和空间对比度见 selection.csv；相关性选帧可能同时改变这些量，不能把质量差异全部解释成信息量差异。',
              '物理损失使用各自 90 帧补集，仅作辅助；补集参与过 100 帧选帧，因此不是独立验证集。',
              '原十个随机子集先按对象平均，再对三个对象等权平均；子集不是独立重复，不宣称统计显著。',
              '固定资产清单：'+run['manifest'], '',
              '完整数值见 metrics.csv、per_object.csv、object_macro.csv；低减高和方法间差值分别见 low_minus_high.csv、method_differences.csv。']
    (output/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    assert len([r for r in rows if r['group'] in ('low','high')])==60
    save(output/'report_complete.json',{'complete':True,'challenge_metric_rows':60,'random_reference_rows':300,
                                        'method_roles':10,'objects':3,'groups':2})
