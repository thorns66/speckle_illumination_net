"""CPU-only grouped views of frozen spinach reconstructions; no inference."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, subprocess, sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('MPLCONFIGDIR','/tmp/spinach_grouped_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/spinach_v5_800_detailed_20260913_run02'
SKILL=Path('/workspace/xyx/.codex/skills/nature-figure/scripts')
sys.path.insert(0,str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment
GROUPS={'A_RL_vs_Mean100':['mean_rl3','taylor_rl3_sqrt','mean800'],
        'B_Taylor100_vs_Mean100':['taylor800','mean800']}
LABELS={'mean_rl3':'Mean-RL3','taylor_rl3_sqrt':'Taylor-RL3-sqrt',
        'mean800':'Mean100 (800 steps)','taylor800':'Taylor100 (800 steps)'}
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
                     'font.size':12,'pdf.fonttype':42,'svg.fonttype':'none'})

def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False))

def export(fig,path,book):
    require_matplotlib_panel_alignment(fig,json_out=str(path)+'.alignment.json',strict=True)
    fig.savefig(str(path)+'.png',dpi=300)
    fig.savefig(str(path)+'.pdf')
    if path.name=='projections':fig.savefig(str(path)+'.svg')
    book.savefig(fig)
    env=dict(os.environ,PYTHONPATH='/tmp/speckle_figure_qa_deps')
    for tool,args in [('audit_pdf_text.py',['--min-pt','5','--json']),
                      ('audit_figure_collisions.py',['--json-out',str(path)+'.collision.json'])]:
        result=subprocess.run([sys.executable,str(SKILL/tool),str(path)+'.pdf',*args],
                              env=env,capture_output=True,text=True)
        Path(str(path)+'.'+tool+'.log').write_text(result.stdout+result.stderr)
        if result.returncode: raise RuntimeError(str(path)+': '+tool+' '+result.stdout[-500:])
    plt.close(fig)

def image(ax,value,scale,title,aspect='equal'):
    ax.imshow(np.clip(value/scale,0,1),vmin=0,vmax=1,cmap='magma',aspect=aspect,interpolation='nearest')
    ax.set_title(title,pad=8);ax.set_axis_off()

def group(out,field,group_id,methods,volumes,scales,mode):
    from matplotlib.backends.backend_pdf import PdfPages
    folder=out/field/group_id/mode;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    n=len(methods);width=5*n
    # Physical XY ratio is fixed; XZ/YZ depth is stretched only for display.
    panel_width=width*(.98-.03)/(n+(n-1)*.08)
    height=(panel_width*1029/1421)*(5/3)*(1+2*.32/3)/(.9-.045)
    chosen={m:scales[m] if mode=='own' else max(scales[k] for k in methods) for m in methods}
    write(folder/'display_scales.json',{'scale':chosen,'mode':mode,'gamma':1,'percentile':99.9,
        'whole_volume_not_per_layer':True,'roi_yx':[343,735,490,882],
        'clipped_fraction':{m:float((volumes[m]>chosen[m]).mean()) for m in methods}})
    with PdfPages(folder/'atlas.pdf') as book:
        fig,axes=plt.subplots(3,n,figsize=(width,height),gridspec_kw={'height_ratios':[3,1,1]})
        fig.subplots_adjust(left=.03,right=.98,bottom=.045,top=.9,wspace=.08,hspace=.32)
        for c,m in enumerate(methods):
            v=volumes[m]
            for r,im in enumerate([v.max(0),v.max(1),v.max(2)]):
                image(axes[r,c],im,chosen[m],LABELS[m]+' / '+['XY','XZ','YZ'][r], 'equal' if r==0 else 'auto')
        fig.suptitle(f'Field {field} | subset 01 | {mode} whole-volume scale',fontsize=14,y=.985)
        export(fig,folder/'projections',book)
        for layer in range(10):
            layer_height=panel_width*(1029/1421+1)*(1+.23/2)/(.91-.04)
            fig,axes=plt.subplots(2,n,figsize=(width,layer_height),gridspec_kw={'height_ratios':[1029/1421,1]})
            fig.subplots_adjust(left=.03,right=.98,bottom=.04,top=.91,wspace=.08,hspace=.23)
            for c,m in enumerate(methods):
                v=volumes[m]
                image(axes[0,c],v[layer],chosen[m],LABELS[m]+' / full field')
                image(axes[1,c],v[layer,343:735,490:882],chosen[m],LABELS[m]+' / fixed ROI')
            fig.suptitle(f'Field {field} | depth {10+10*layer} um | {mode} scale, fixed across all depths',fontsize=14,y=.985)
            export(fig,folder/f'layer_{10+10*layer:03d}um',book)
        fig,ax=plt.subplots(figsize=(10,5.5));fig.subplots_adjust(left=.12,right=.7,bottom=.15,top=.88)
        for m in methods:
            p=volumes[m].sum((1,2),dtype=np.float64);p/=max(p.sum(),1e-30)
            ax.plot(np.arange(10,101,10),p,'.-',label=LABELS[m])
        ax.set_xlabel('Depth (um)');ax.set_ylabel('Fraction of total volume mass')
        ax.set_title(f'Field {field}: native-depth distribution')
        ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),fontsize=10)
        export(fig,folder/'depth_profile',book)
    write(folder/'complete.json',{'complete':True,'pages':12,'methods':methods})
    print(f'COMPLETE {field} {group_id} {mode}',flush=True)

def main(out):
    if out.exists():
        assert json.loads((out/'manifest.json').read_text())['groups']==GROUPS
    else:out.mkdir(parents=True)
    contract={'question':'Compare RL priors to Mean100 separately from Taylor100 versus Mean100.',
              'archetype':'image plate + native-depth quantification','backend':'Python',
              'reuse':'structural adaptation of existing volume display; method groups explicitly selected by user',
              'groups':GROUPS,'subset':1,'fields':['45','55'],'no_GT':True,'training_fields':True,
              'network_checkpoint':'final800, E3 + mean gradient budget <=100%; 100 is not frame count',
              'display':'linear whole-volume p999; no layer-wise scaling; XY native aspect; XZ/YZ stretched',
              'exports':'300 dpi wide-screen PNG, editable PDF; projection SVG; no journal submission claim',
              'source':str(SOURCE),'hashes':{}}
    write(out/'manifest.json',contract)
    for field in ['45','55']:
        volumes={m:np.load(SOURCE/field/'subset_01'/m/'reconstruction.npy') for m in set(sum(GROUPS.values(),[]))}
        for m,v in volumes.items():
            assert v.shape==(10,1029,1421) and np.isfinite(v).all() and v.min()>=0
            contract['hashes'][field+'/'+m]=hashlib.sha256(v.tobytes()).hexdigest()
        scales={m:max(float(np.quantile(v,.999)),1e-30) for m,v in volumes.items()}
        for gid,methods in GROUPS.items():
            for mode in (['own'] if gid.startswith('A') else ['own','shared']):
                group(out,field,gid,methods,volumes,scales,mode)
    write(out/'manifest.json',contract)
    lines=['# 菠菜根分组对比图册','',
      'A组：Mean-RL3／Taylor-RL3-sqrt／Mean100。B组：Taylor100／Mean100。',
      'Mean100、Taylor100分别指两种重建基础的E3＋mean≤100%网络，均使用本次800步权重；不是100帧输入。',
      '45、55均为训练视场，无GT。复用固定subset_01结果，只重绘，无新推理。',
      '每幅图只有两或三种方法。各原生层同时提供全幅和相同392像素ROI，不逐层归一化；B组另有共同尺度版本。',
      '', '|视场|A组三方法|B组两网络|B组共同尺度|','|---|---|---|---|']
    for f in ['45','55']:
        paths=[f'{f}/A_RL_vs_Mean100/own',f'{f}/B_Taylor100_vs_Mean100/own',f'{f}/B_Taylor100_vs_Mean100/shared']
        lines.append('|'+f+'|'+'|'.join(f'[投影]({p}/projections.png) · [完整图册PDF]({p}/atlas.pdf)' for p in paths)+'|')
    lines+=['','每本PDF含投影、10个原生层及ROI、深度曲线，共12页。原始重建体和旧六方法图不变。',
            '技能QA：使用固定尺度、等比例XY和逐图对齐/碰撞/字体检查；宽幅300 dpi便于屏幕检查，不声称期刊最终版式。']
    (out/'README_ZH.md').write_text('\n'.join(lines))
    write(out/'complete.json',{'complete':True,'groups':2,'atlases':6,'pages':72,'gpu_used':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();main(a.output)
