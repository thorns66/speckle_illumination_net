"""Plot saved reconstructions directly: GT/baseline/E1/E2/E3, best and final."""
from pathlib import Path
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/parallel_validation_scale_cov_mean_20260907'
DEST=RUN/'analysis/actual_reconstruction_comparison'
METHODS=['baseline','e1','e2','e3']
NAMES=['GT 真值','原 baseline','E1 亮度尺度','E2 相关方差','E3 均值与方差']
FONT=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name()
plt.rcParams.update({'font.family':FONT,'font.size':11,'axes.unicode_minus':False})
CASES=[('P07_subset_01','P07 测试对象 / 子集 01'),('T01_subset_01','T01 测试对象 / 子集 01'),
       ('T02_subset_01','T02 测试对象 / 子集 01'),('points_z060_r01','60 µm 点与点对 / 第一次采集'),
       ('lines_z060_r01','60 µm 线条与断口 / 第一次采集'),('axial_pairs_r01','轴向点对 / 第一次采集')]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_case(case):
    gt_path=(ROOT/'data/speckle_data_now'/case.split('_subset_')[0]/'prepared.mat' if '_subset_' in case
             else ROOT/'outputs/priority_validation_20260907/generated'/case/'prepared.mat')
    with h5py.File(gt_path,'r') as f:truth=np.asarray(f['ground_truth'],np.float32).transpose(0,2,1).copy()
    sigma=float(np.load(RUN/'illumination/calibration_moments.npz')['sigma'])
    values={'best':[truth*sigma],'final':[truth*sigma]};records=[];expected_indices=None
    for role in ['best','final']:
        for method in METHODS:
            folder=RUN/'evaluation'/method/role/case
            record=json.loads((folder/'complete.json').read_text())
            pred_path=folder/'reconstruction.npy';pred=np.load(pred_path)
            assert record['complete'] and digest(pred_path)==record['prediction_sha256']
            assert pred.shape==truth.shape==(10,260,260) and np.isfinite(pred).all() and (pred>=0).all()
            indices=record['input_indices']
            if expected_indices is None:expected_indices=indices
            assert indices==expected_indices
            values[role].append(pred)
            records.append({'method':method,'role':role,'step':record['weight_step'],
                 'prediction_path':str(pred_path),'prediction_sha256':record['prediction_sha256'],
                 'checkpoint_sha256':record['checkpoint_sha256'],'input_indices':indices,
                 'volume_max':float(pred.max()),'volume_sum':float(pred.sum())})
    return values,records,{'path':str(gt_path),'sha256':digest(gt_path),'density_to_fixed_unit':sigma}

def projections(case,title,values,records,display):
    fig=plt.figure(figsize=(16,12),facecolor='white')
    gs=fig.add_gridspec(4,5,height_ratios=[2.5,1,2.5,1],hspace=.44,wspace=.10,
                       left=.055,right=.975,bottom=.075,top=.88)
    shared_max=max(float(v.max()) for volumes in values.values() for v in volumes)
    for ri,role in enumerate(['best','final']):
        for col,raw in enumerate(values[role]):
            scale=max(float(raw.max()),1e-30) if display=='shape' else max(shared_max,1e-30)
            shown=raw/scale
            ax=fig.add_subplot(gs[2*ri,col])
            ax.imshow(shown.max(0),cmap='magma',vmin=0,vmax=1,interpolation='nearest')
            if col:
                step=next(r['step'] for r in records if r['method']==METHODS[col-1] and r['role']==role)
                ax.set_title(NAMES[col]+f'\n{role} · 第 {step} 步',fontsize=12,pad=6)
            else:ax.set_title('GT 真值\n同一目标',fontsize=12,pad=6)
            ax.set_xticks([]);ax.set_yticks([])
            if col==0:ax.set_ylabel(f'{role}\n从上往下看（XY）',fontsize=11,labelpad=9)
            ax=fig.add_subplot(gs[2*ri+1,col])
            ax.imshow(shown.max(1),cmap='magma',vmin=0,vmax=1,interpolation='nearest',
                      aspect='auto',extent=[0,292.5,105,5])
            ax.set_yticks([10,50,100]);ax.set_xticks([0,150,292.5]);ax.tick_params(labelsize=9)
            if col==0:ax.set_ylabel('从侧面看（XZ）\n深度 / µm',fontsize=11)
            else:ax.set_yticklabels([])
            ax.set_xlabel('X / µm',fontsize=10,labelpad=1)
    fig.suptitle(title+'：实际重建对比',fontsize=19,y=.985)
    explanation=('看结构：每份完整三维体只除以自己的一个最大值，十层共用；不能据此比较绝对亮度。'
                 if display=='shape' else
                 '看亮度：所有方法、best/final 和固定单位 GT 共用同一显示上限；较暗结果没有单独调亮。')
    fig.text(.5,.945,explanation,ha='center',fontsize=12)
    fig.text(.5,.032,'同一组十帧输入；直接读取保存的预测；不平均重复、不平滑、不逐层调亮度。'
             +' 上视图和侧视图均为最大强度投影。',ha='center',fontsize=10)
    path=DEST/f'{case}_{display}.png';fig.savefig(path,dpi=160);plt.close(fig)
    return path

def slices(case,title,role,volumes,records):
    fig,axes=plt.subplots(5,10,figsize=(21,11),gridspec_kw={'wspace':.04,'hspace':.08})
    for row,volume in enumerate(volumes):
        shown=volume/max(float(volume.max()),1e-30)
        for z in range(10):
            ax=axes[row,z];ax.imshow(shown[z],cmap='magma',vmin=0,vmax=1,interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(f'{10+z*10} µm',fontsize=12)
            if z==0:
                text=NAMES[row]
                if row:text+='\n第 '+str(next(r['step'] for r in records if r['method']==METHODS[row-1] and r['role']==role))+' 步'
                ax.set_ylabel(text,fontsize=12)
    fig.suptitle(title+f'：{role} 的十个实际深度层',fontsize=19,y=.98)
    fig.text(.5,.025,'每行仅按整卷最大值归一化，十个深度层保持同一亮度比例；不使用投影、不逐层调亮。',ha='center',fontsize=12)
    fig.subplots_adjust(left=.075,right=.99,bottom=.065,top=.93)
    path=DEST/f'{case}_{role}_layers.png';fig.savefig(path,dpi=150);plt.close(fig);return path

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    manifest={'case_selection':'First subset of each held-out test object; first acquisition of fixed local scenes, no selection by model scores',
              'display_modes':{'shape':'one positive maximum per entire 3-D volume, shared across every Z layer',
                               'shared':'one maximum shared by all methods and both checkpoint roles, GT in fixed calibrated units'},
              'raw_predictions_modified':False,'cases':[]}
    gallery=['# 三组实验与原 baseline：实际重建图','',
      '每张总览固定五列：GT、原 baseline、E1、E2、E3；上半部分是 best，下半部分是 final。',
      'XY 是从上往下看的最大强度投影，XZ 是从侧面看的最大强度投影。十层图则直接展示 10–100 µm 的实际切片。','',
      '“看结构”只对整份三维体用一个显示比例；“看亮度”所有方法和 best/final 使用同一比例。两者均不逐层调亮度，原始预测没有改动。',
      '测试对象固定选 P07、T01、T02 的第 01 子集；局部场景固定选第一次采集，不按网络效果挑图。','']
    for case,title in CASES:
        values,records,gt=load_case(case);files=[]
        for display in ['shape','shared']:files.append(projections(case,title,values,records,display))
        if '_subset_' in case:
            for role in ['best','final']:files.append(slices(case,title,role,values[role],records))
        manifest['cases'].append({'case_id':case,'truth':gt,'predictions':records,
                     'figures':[{ 'file':p.name,'sha256':digest(p)} for p in files]})
        gallery.extend([f'## {title}','',f'![结构对比]({case}_shape.png)','',
             f'[统一亮度对比]({case}_shared.png)'+(f' · [best 十层切片]({case}_best_layers.png) · [final 十层切片]({case}_final_layers.png)' if '_subset_' in case else ''),''])
        print('Saved actual reconstruction figures: '+case,flush=True)
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    (DEST/'README_ZH.md').write_text('\n'.join(gallery)+'\n')

if __name__=='__main__':main()
