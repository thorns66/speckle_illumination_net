"""CPU report for the fixed three-method real-data exploratory comparison."""
from __future__ import annotations
import csv,json,math
from pathlib import Path
import h5py,numpy as np,tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS=('mean_rl3','taylor_rl3_sqrt','e3_mean100')

def mat(path,name,volume=False):
    with h5py.File(path,'r') as h:value=np.asarray(h[name],dtype=np.float32)
    return value.transpose(0,2,1).copy() if volume else value.T.copy()

def scale_nrmse(pred,target):
    p=pred.astype(np.float64);t=target.astype(np.float64);gain=np.vdot(p,t).real/max(np.vdot(p,p).real,1e-30)
    return float(gain),float(np.linalg.norm(gain*p-t)/max(np.linalg.norm(t),1e-30))

def shape_rmse(pred,target,log=False):
    p=pred.astype(np.float64)/max(float(pred.mean()),1e-30);t=target.astype(np.float64)/max(float(target.mean()),1e-30)
    if log:p=np.log(p+1e-6);t=np.log(t+1e-6)
    return float(np.sqrt(np.mean((p-t)**2)))

def corr(a,b):
    x=a.astype(np.float64).ravel();y=b.astype(np.float64).ravel();x-=x.mean();y-=y.mean()
    return float(np.dot(x,y)/max(np.linalg.norm(x)*np.linalg.norm(y),1e-30))

def write_csv(path,rows):
    columns=list(dict.fromkeys(k for r in rows for k in r));
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)

def show(ax,image,scale,title,aspect='equal'):
    ax.imshow(np.clip(image/scale,0,1),cmap='magma',vmin=0,vmax=1,interpolation='nearest',aspect=aspect)
    ax.set_title(title,fontsize=9);ax.set_xticks([]);ax.set_yticks([])

def figures(out,volumes,frames,mean,var):
    f=out/'figures';f.mkdir(exist_ok=True)
    fig,axs=plt.subplots(2,5,figsize=(16,7),constrained_layout=True)
    for ax,image,index in zip(axs.ravel(),frames,range(10)):
        show(ax,image,max(float(np.quantile(image,.999)),1e-30),f'Frame {index+1} · own p99.9')
    fig.savefig(f/'selected_input_frames.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    show(axs[0],mean,max(float(np.quantile(mean,.999)),1e-30),'10-frame mean · global p99.9')
    show(axs[1],var,max(float(np.quantile(var,.999)),1e-30),'10-frame variance · global p99.9')
    fig.savefig(f/'input_mean_variance.png',dpi=150);plt.close(fig)
    for policy in ('shape_max','shape_p999','shared_p999'):
        own={m:max(float(v.max() if policy=='shape_max' else np.quantile(v,.999)),1e-30) for m,v in volumes.items()}
        if policy=='shared_p999':
            common=max(own.values());own={m:common for m in METHODS}
        fig,axs=plt.subplots(3,3,figsize=(14,11),constrained_layout=True)
        for col,m in enumerate(METHODS):
            v=volumes[m];show(axs[0,col],v.max(0),own[m],m+' XY')
            show(axs[1,col],v.max(1),own[m],m+' XZ','auto');show(axs[2,col],v.max(2),own[m],m+' YZ','auto')
        fig.suptitle(policy+'; one scale per complete volume, never per layer')
        fig.savefig(f/f'projections_{policy}.png',dpi=150);plt.close(fig)
        fig,axs=plt.subplots(10,3,figsize=(12,32),constrained_layout=True)
        for col,m in enumerate(METHODS):
            for z in range(10):
                show(axs[z,col],volumes[m][z],own[m],f'{m} · {10*(z+1)} um')
        fig.savefig(f/f'native_layers_{policy}.png',dpi=100);plt.close(fig)
    rois={'center':(7*49,14*49,11*49,18*49),'upper_left':(2*49,9*49,2*49,9*49),
          'lower_right':(12*49,19*49,20*49,27*49)}
    for name,(y0,y1,x0,x1) in rois.items():
        fig,axs=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
        for ax,m in zip(axs,METHODS):
            v=volumes[m][:,y0:y1,x0:x1];s=max(float(np.quantile(v,.999)),1e-30);show(ax,v.max(0),s,m+' · '+name)
        fig.savefig(f/f'roi_{name}_xy.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
    for m,v in volumes.items():
        p=v.sum((1,2),dtype=np.float64);p/=max(p.sum(),1e-30);ax.plot(range(10,101,10),p,'.-',label=m)
    ax.set_xlabel('Depth (um; PSF coordinate)');ax.set_ylabel('Volume mass fraction');ax.grid(alpha=.25);ax.legend()
    fig.savefig(f/'axial_mass_profiles.png',dpi=160);plt.close(fig)

def report(output,manifest):
    out=Path(output);cfg=json.loads(Path(manifest).read_text());paths=cfg['output_mat']
    volumes={'mean_rl3':mat(paths['mean'],'reconstruction_raw',True),
             'taylor_rl3_sqrt':mat(paths['taylor'],'reconstruction_sqrt',True),
             'e3_mean100':mat(cfg['network_final_mat'],'reconstruction',True)}
    predictions={'mean_rl3':(mat(paths['mean'],'predicted_mean'),mat(paths['mean'],'predicted_variance')),
                 'taylor_rl3_sqrt':(mat(paths['taylor'],'predicted_mean'),mat(paths['taylor'],'predicted_variance')),
                 'e3_mean100':(mat(cfg['network_final_mat'],'predicted_mean'),mat(cfg['network_final_mat'],'predicted_variance'))}
    frames=np.stack([tifffile.imread(p).astype(np.float32)/255 for p in cfg['selected_files']])
    hold=np.stack([tifffile.imread(p).astype(np.float32)/255 for p in cfg['holdout_files']])
    targets={'input10':(frames.mean(0,dtype=np.float64).astype(np.float32),frames.var(0,ddof=1,dtype=np.float64).astype(np.float32)),
             'holdout90':(hold.mean(0,dtype=np.float64).astype(np.float32),hold.var(0,ddof=1,dtype=np.float64).astype(np.float32))}
    rows=[]
    for m,(pm,pv) in predictions.items():
        v=volumes[m];mass=v.sum((1,2),dtype=np.float64);fraction=mass/max(mass.sum(),1e-30)
        dz=np.arange(10,101,10);entropy=float(-(fraction*np.log(fraction+1e-30)).sum())
        rough=float((np.abs(np.diff(v,axis=1)).sum()+np.abs(np.diff(v,axis=2)).sum())/max(v.sum(),1e-30))
        for target,(tm,tv) in targets.items():
            mg,mn=scale_nrmse(pm,tm);vg,vn=scale_nrmse(pv,tv)
            rows.append({'method':m,'target':target,'mean_scale_aligned_nrmse':mn,'mean_gain':mg,
              'mean_shape_rmse':shape_rmse(pm,tm),'mean_pearson':corr(pm,tm),
              'variance_scale_aligned_nrmse':vn,'variance_gain':vg,
              'variance_log_shape_rmse':shape_rmse(pv,tv,True),'variance_pearson':corr(pv,tv),
              'volume_sum':float(v.sum(dtype=np.float64)),'volume_max':float(v.max()),
              'axial_centroid_um':float(np.dot(dz,fraction)),'axial_peak_um':int(dz[np.argmax(fraction)]),
              'axial_entropy':entropy,'normalized_lateral_total_variation':rough})
    exported=[];root=out/'volumes'
    for m,v in volumes.items():
        d=root/m;d.mkdir(parents=True,exist_ok=True);np.save(d/'reconstruction.npy',v)
        tifffile.imwrite(d/'reconstruction_float.tif',v,metadata={'axes':'ZYX','z_um':list(range(10,101,10))})
        exported.append({'method':m,'shape':list(v.shape),'npy':str(d/'reconstruction.npy'),'tiff':str(d/'reconstruction_float.tif')})
    write_csv(out/'physics_metrics.csv',rows);figures(out,volumes,frames,*targets['input10'])
    primary={r['method']:r for r in rows if r['target']=='holdout90'}
    lines=['# 菠菜根真实数据探索性推理','',
      '本轮使用现有逐帧归一化、8位、已校正 TIFF；没有 GT，不宣称绝对重建准确率或分辨率提升。',
      f"固定输入帧：{cfg['input_indices']}。PSF 深度为 10–100 μm 十层。E3＋mean≤100% 使用第 400 步 final。",'',
      '| 方法 | 90帧均值一致性 NRMSE↓ | 90帧方差一致性 NRMSE↓ | 方差相关↑ | 轴向质心 μm | 横向归一化TV |',
      '|---|---:|---:|---:|---:|---:|']
    for m in METHODS:
        r=primary[m];lines.append(f"| {m} | {r['mean_scale_aligned_nrmse']:.4f} | {r['variance_scale_aligned_nrmse']:.4f} | {r['variance_pearson']:.4f} | {r['axial_centroid_um']:.2f} | {r['normalized_lateral_total_variation']:.4f} |")
    lines += ['', '这些指标只是前向统计一致性。数值更低不等于真实结构或真实分辨率更高；锐度更高也可能来自伪影。',
      '上游矫正对每帧独立除以最大值并保存为 uint8，因此这不是与仿真 retained-scale physics 流严格匹配的实验。','',
      '## 结果入口','',
      '- 三种浮点三维体：`volumes/`（NPY 和多页浮点 TIFF）。',
      '- 清晰投影：`figures/projections_shape_p999.png`；共同尺度：`figures/projections_shared_p999.png`。',
      '- 十个原生层：`figures/native_layers_shape_p999.png`；固定局部区域与轴向曲线也在 `figures/`。',
      '- 全部指标：`physics_metrics.csv`；运行合同：`manifest.json`。','',
      f"网络推理方式：{json.loads(Path(cfg['network_record']).read_text())['mode']}。"]
    (out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    (out/'report_complete.json').write_text(json.dumps({'complete':True,'methods':list(METHODS),
      'volumes':exported,'figures':len(list((out/'figures').glob('*.png'))),'metric_rows':len(rows)},indent=2,ensure_ascii=False))

