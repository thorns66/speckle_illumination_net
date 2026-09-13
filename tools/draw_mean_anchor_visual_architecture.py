"""Image-led architecture plate; all imagery is sourced from saved experiments."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch, FancyArrowPatch
from matplotlib.transforms import Affine2D
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.experiment_paths import next_experiment_path
sys.path.insert(0, '/workspace/xyx/.codex/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment

SOURCE = ROOT/'outputs/spinach_root_v5_taylor600_compare_20260911_run01'
C = {'mean':'#266EB1','var':'#CF7029','set':'#578638','net':'#7955AA',
     'ink':'#172A3C','muted':'#56697A','border':'#BDC9D3','gray':'#EDF1F4'}
SCALE = 1.0


def txt(ax,x,y,s,size=8.3,color='ink',ha='center',weight='normal',**kwargs):
    return ax.text(x,y,s,ha=ha,va='center',fontsize=size*SCALE,
                   color=C.get(color,color),weight=weight,linespacing=1.35,**kwargs)


def pale(family,factor=.10):
    return tuple(1-factor*(1-v) for v in to_rgb(C.get(family,family)))


def rect(ax,x,y,w,h,family='border',fill=None,lw=.7):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=.9',
        facecolor=fill if fill is not None else pale(family),
        edgecolor=C.get(family,family),linewidth=lw*SCALE,zorder=2)
    ax.add_patch(p)
    return p


def line(ax,points,color='muted',lw=.85,dash=False,head=True):
    p=np.array(points)
    color=C.get(color,color)
    if len(p)>2 or not head:
        part=p[:-1] if head else p
        ax.plot(part[:,0],part[:,1],color=color,lw=lw*SCALE,
                ls=(0,(3,2)) if dash else '-',zorder=1)
    if head:
        ax.add_patch(FancyArrowPatch(p[-2],p[-1],arrowstyle='-|>',
            mutation_scale=7*SCALE,lw=lw*SCALE,color=color,
            linestyle=(0,(3,2)) if dash else '-',shrinkA=0,shrinkB=0,zorder=1))


def tile(ax,array,x,y,w,h,color=None,limit=None,gamma=.65):
    limit=float(np.quantile(np.abs(array),.999)) if limit is None else limit
    n=np.clip(np.abs(array)/max(limit,1e-30),0,1)**gamma
    if color is None:
        rgb=np.repeat(n[...,None],3,axis=2)
    else:
        hue=np.array(to_rgb(C[color]))*.65+.35
        rgb=.022+n[...,None]*(hue-.022)
    ax.imshow(rgb,extent=(x,x+w,y,y+h),origin='upper',interpolation='antialiased',zorder=3,aspect='auto')
    ax.add_patch(Rectangle((x,y),w,h,fill=False,ec=C[color] if color else C['muted'],lw=.7*SCALE,zorder=4))


def cube(ax,v,x,y,w,h,d,family,limit):
    """Three orthogonal maximum projections on affine cube faces, no ray casting."""
    hue=np.array(to_rgb(C[family]))*.55+.45
    transforms=[(v.max(0),Affine2D.from_values(w,0,0,h,x,y)),
                (v.max(1),Affine2D.from_values(w,0,d,.58*d,x,y+h)),
                (v.max(2).T,Affine2D.from_values(d,.58*d,0,h,x+w,y))]
    for a,t in transforms:
        n=np.clip(a/max(limit,1e-30),0,1)**.50
        rgb=.025+n[...,None]*(hue-.025)
        ax.imshow(rgb,extent=(0,1,0,1),origin='upper',interpolation='antialiased',
                  transform=t+ax.transData,zorder=3,aspect='auto')
    pts=[[(x,y),(x+w,y),(x+w,y+h),(x,y+h),(x,y)],
         [(x,y+h),(x+d,y+h+.58*d),(x+w+d,y+h+.58*d),(x+w,y+h)],
         [(x+w,y),(x+w+d,y+.58*d),(x+w+d,y+h+.58*d)]]
    for p in pts: line(ax,p,color=family,lw=.8,head=False)
    # Redraw cube outlines above image faces.
    for p in pts:
        p=np.array(p);ax.plot(p[:,0],p[:,1],color=C[family],lw=.8*SCALE,zorder=4)


def plates(ax,x,y,h,family,n=5):
    for j in range(n-1,-1,-1):
        dx=j*1.15; dy=j*.42
        verts=[(x+dx,y+dy),(x+dx+2.7,y+dy+.85),
               (x+dx+2.7,y+h+dy+.85),(x+dx,y+h+dy)]
        ax.add_patch(Polygon(verts,closed=True,facecolor=pale(family,.13+j*.06),
                            edgecolor=C[family],lw=.55*SCALE,zorder=3))


def setup(ax,key,title,w,h):
    ax.set_xlim(0,w);ax.set_ylim(0,h);ax.axis('off');ax.set_label(key)
    rect(ax,.3,.3,w-.6,h-.6,'border',fill='white',lw=.7).set_zorder(-5)
    txt(ax,2.2,h-4,key,size=11,weight='bold',ha='left')
    txt(ax,8,h-4,title,size=10,weight='bold',ha='left')


def preprocessing(ax,data):
    setup(ax,'a','Physics preprocessing',57,94)
    txt(ax,28.5,84.5,'Same 10 input frames for all branches',size=7.5,color='muted')
    for j in range(3):
        tile(ax,data['frames'][j],3+1.1*j,43+1.6*j,10,14,limit=data['frame_limit'])
    txt(ax,8.8,36,'Light-field\nimage stack',size=7.2)
    txt(ax,8.8,26.2,r'$N=10$',size=10)
    for y,col,name,arr,formula in (
        (65,'mean','Mean',data['mu'],r'$\mu_{10}$'),
        (43,'var','Variance',data['var'],r'$v_{10}$'),
        (18,'set','Residuals',data['frames'][0]-data['mu'],r'$I_i-\mu_{10}$')):
        line(ax,[(15,50),(18,50),(18,y+5),(21,y+5)],col)
        tile(ax,arr,21,y,12,12,color=None,gamma=.55)
        txt(ax,27,y+15.3,name,size=8.3,color=col,weight='bold')
        txt(ax,27,y-4,formula,size=10)
        line(ax,[(33,y+6),(37,y+6)],col)
        rect(ax,37,y,17,12,col)
        txt(ax,45.5,y+8.5,'RL3' if col!='set' else 'Set',size=9,color=col,weight='bold')
        txt(ax,45.5,y+3.4,{'mean':'PSF H','var':'H² → sqrt','set':'10 frames'}[col],size=7.1)
    txt(ax,28.5,5.8,'Unbiased variance · three RL iterations',size=7.0,color='muted')


def network(ax,data):
    setup(ax,'b','Three-branch residual reconstruction network',173,94)
    for y,col,key in ((64,'mean','mean'),(40,'var','varvol')):
        cube(ax,data[key],3,y,10,12,3,col,data['limits'][key])
        txt(ax,9.5,y-4, 'Mean' if col=='mean' else 'Taylor',size=8.4,color=col,weight='bold')
        txt(ax,9.5,y-8,'Mean base' if col=='mean' else 'sqrt features',size=7,color=col)
        line(ax,[(17,y+6),(22,y+6)],col)
        for j,x in enumerate((24,39,54)):
            h=(12,9,6)[j]
            plates(ax,x,y+6-h/2,h,col,n=5)
            txt(ax,x+4,y+6-h/2-3,str((16,32,64)[j]),size=7.6,color=col)
            if j<2:line(ax,[(x+8.3,y+6),(x+14,y+6)],col,lw=.7)
        # All three pyramid levels feed the bus, not just the deepest level.
        line(ax,[(61.3,y+6),(63,y+6)],col,lw=.65,head=False)
        for port in (33,48,63):
            line(ax,[(port,y+6),(port,79 if col=='mean' else 33)],col,lw=.65,head=False)
        if col=='mean':
            line(ax,[(33,79),(120,79)],col,lw=.65,head=False)
        else:
            line(ax,[(33,33),(69,33),(69,77),(120,77)],col,lw=.65,head=False)
    txt(ax,45,83.5,'3D encoders · RMS normalized',size=7.5,color='muted')
    txt(ax,98,85,'Matched scales: 1 / ½ / ¼',size=7.3,color='net')
    for j,x in enumerate((74,92,110)):
        line(ax,[(x+5,79),(x+5,76)],'mean',lw=.65)
        line(ax,[(x+10,77),(x+10,76)],'var',lw=.65)
        rect(ax,x,60,15,16,'net')
        txt(ax,x+7.5,72,'Fusion '+str(j),size=7.7,color='net',weight='bold')
        txt(ax,x+7.5,67,'Mean +\nTaylor',size=7.1)
        txt(ax,x+7.5,62.5,'1×1×1',size=7.1)
        line(ax,[(x+7.5,60),(x+7.5,51)],'net')
        rect(ax,x,34,15,17,'net')
        txt(ax,x+7.5,46.5,'Gated Set',size=7.3,color='net',weight='bold')
        txt(ax,x+7.5,39.7,'Add gated\nSet features',size=7.0)
        line(ax,[(x+3,28),(x+3,34)],'set')
    # Set extraction uses measured residual frames and schematic feature plates.
    for j in (2,1,0):tile(ax,data['frames'][j]-data['mu'],3+j,10+j*.8,9,10,color='set',gamma=.6)
    txt(ax,9,5.2,'10 residuals',size=7,color='set')
    for x,w,title,sub in ((20,16,'2D CNN','Per frame'),
                           (40,17,'Pooling','Mean / std'),
                           (61,15,'Lift + z','2D → 3D')):
        rect(ax,x,9,w,14,'set')
        txt(ax,x+w/2,18.6,title,size=7.3,color='set',weight='bold')
        txt(ax,x+w/2,12.5,sub,size=7)
    line(ax,[(14,16),(20,16)],'set')
    line(ax,[(36,16),(40,16)],'set')
    line(ax,[(57,16),(61,16)],'set')
    line(ax,[(76,16),(119,16),(119,28),(77,28)],'set',head=False)
    txt(ax,102,10,'Set channels: 8 / 16 / 32',size=7,color='set')
    txt(ax,48,4,'Order invariant · 2D → 3D feature lifting',size=7,color='set')
    # Deep-to-shallow decoder staircase, one fused scale at each level.
    rect(ax,132,33,38,46,'net',fill=pale('net',.035)).set_zorder(0)
    txt(ax,151,74.4,'Residual 3D decoder',size=8,color='net',weight='bold')
    for x,y,h,n in ((135,38,7,64),(146,47,10,32),(157,57,12,16)):
        plates(ax,x,y,h,'net',n=4)
        txt(ax,x+3,y-2.6,str(n),size=7.3,color='net')
    line(ax,[(142,42),(145,42),(145,52),(146,52)],'net')
    line(ax,[(153,52),(156,52),(156,63),(157,63)],'net')
    # The three named fused outputs enter their corresponding decoder scale.
    for j,end in enumerate(((157,63),(146,52),(135,42))):
        x=(74,92,110)[j]+12
        lane=55+j*1.5
        line(ax,[(x,51),(x,lane),(128+j,lane),(128+j,end[1]),end],'net',lw=.65)
    txt(ax,151,28.5,'1×1×1 head',size=7.8,color='net')
    line(ax,[(165,63),(168,63),(168,29),(161,29)],'net')
    line(ax,[(151,26),(151,22)],'net')
    # Residual is not visualized as measured activations: colored feature planes only.
    plates(ax,144,9,10,'net',n=5)
    txt(ax,157,14,r'$r_\theta$',size=12,color='net')
    txt(ax,151,4.8,'Signed residual → d',size=7.3,color='net')


def physics(ax,data):
    setup(ax,'c','Self-supervised physics',76,61)
    txt(ax,18,50.8,'From output q, a',size=7.6,color='net')
    txt(ax,63.5,50.8,'90 target frames',size=7.6,color='muted')
    for y,col,op,arr,target in ((33,'mean','H',data['target_mu'],r'$\mu_{90}$'),
                               (14,'var','H²',data['target_var'],r'$v_{90}$')):
        cube(ax,data['out'],3,y,9,10,2,'net',data['limits']['out'])
        txt(ax,17.5,y+11,r'$q$' if col=='mean' else r'$q^2$',size=10,color=col)
        line(ax,[(15,y+5),(20,y+5)],col,dash=True)
        rect(ax,20,y,13,10,col)
        txt(ax,26.5,y+5,op,size=12,color=col)
        line(ax,[(33,y+5),(38,y+5)],col,dash=True)
        rect(ax,38,y,15,10,col)
        txt(ax,45.5,y+5,'Mean loss' if col=='mean' else 'Var. loss',size=7.5,color=col)
        tile(ax,arr,61,y-1,11,12,gamma=.55)
        line(ax,[(61,y+5),(53,y+5)],col,dash=True)
        txt(ax,66.5,y-4.5,target,size=10)
    txt(ax,35.5,5.6,'+ gradient-budgeted mean shape + TV',size=7.4,color='muted')


def reconstruction(ax,data):
    setup(ax,'d','Mean anchor + learned correction + E3 intensity calibration',154,61)
    txt(ax,77,50.8,'Mean-RL3 carries the reconstruction base; all three branches predict its correction',size=7.7,color='muted')
    cube(ax,data['mean'],3,28,15,14,4,'mean',data['limits']['mean'])
    txt(ax,11,23.6,r'$g_m$',size=11,color='mean')
    line(ax,[(22,35),(27,35)],'mean',lw=1.6)
    rect(ax,27,27,22,17,'mean')
    txt(ax,38,39,'Mean anchor',size=8.2,color='mean',weight='bold')
    txt(ax,38,32,r'$A=\beta_mg_m$',size=11,color='mean')
    line(ax,[(49,35),(55,35)],'mean',lw=1.6)
    rect(ax,55,27,25,17,'net')
    txt(ax,67.5,39,'Positive map',size=7.7,weight='bold',color='net')
    txt(ax,67.5,32,r'$u=P(A,r_\theta)$',size=10.5)
    txt(ax,67.5,20.5,r'$r_\theta$ from b',size=10,color='net')
    line(ax,[(67.5,23),(67.5,27)],'net')
    line(ax,[(80,35),(86,35)],'net',lw=1.2)
    rect(ax,86,27,27,17,'net')
    txt(ax,99.5,39,'E3 calibration',size=7.7,color='net',weight='bold')
    txt(ax,99.5,32,r'$q=u/\sum u$',size=11)
    txt(ax,99.5,20.5,r'$a$ from $\mu_{10}$',size=10,color='mean')
    line(ax,[(99.5,23),(99.5,27)],'mean')
    line(ax,[(113,35),(120,35)],'net',lw=1.3)
    cube(ax,data['out'],121,25,22,21,6,'net',data['limits']['out'])
    txt(ax,134,18,r'$\hat g(x,y,z)=a q$',size=11,color='net')
    txt(ax,133,10.8,'3D reconstruction',size=8,color='net',weight='bold')
    txt(ax,40,12,'Zero correction preserves the Mean structure',size=7.6,color='mean')
    txt(ax,65,5.2,'10 depth planes · z = 10–100 μm · whole-volume brightness calibration',size=7.5,color='muted')


def load_data():
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    files=[Path(p) for p in manifest['selected_files']]
    frames=np.stack([tifffile.imread(p) for p in files]).astype(np.float32)
    data={'frames':frames, 'frame_limit':float(np.quantile(frames,.999))}
    for key,mkey in (('mu','mean_tiff'),('var','variance_tiff'),
                     ('target_mu','holdout_mean_tiff'),('target_var','holdout_variance_tiff')):
        path=Path(manifest[mkey]); data[key]=tifffile.imread(path).astype(np.float32);files.append(path)
    for key,name in (('mean','mean_rl3'),('varvol','taylor_rl3_sqrt'),('out','specified_old_mean_anchor_400')):
        path=SOURCE/'volumes'/f'{name}.npy';data[key]=np.load(path);files.append(path)
        assert data[key].shape[0]==10 and np.isfinite(data[key]).all()
    data['limits']={k:float(np.quantile(data[k],.999)) for k in ('mean','varvol','out')}
    return data,files,manifest


def draw(output,data,width):
    global SCALE
    SCALE=width/240
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':8.3*SCALE,'svg.fonttype':'none','pdf.fonttype':42,
        'mathtext.fontset':'dejavusans','savefig.facecolor':'white'})
    fig=plt.figure(figsize=(width/25.4,164*SCALE/25.4))
    gs=fig.add_gridspec(2,12,left=3/240,right=237/240,bottom=3/164,top=159/164,
        wspace=.12,hspace=.055,height_ratios=[94,61])
    axes=[fig.add_subplot(gs[0,:3]),fig.add_subplot(gs[0,3:]),
          fig.add_subplot(gs[1,:4]),fig.add_subplot(gs[1,4:])]
    preprocessing(axes[0],data);network(axes[1],data)
    physics(axes[2],data);reconstruction(axes[3],data)
    fig.canvas.draw()
    stem=output/('mean_anchor_visual' if width==183 else 'mean_anchor_visual_master')
    require_matplotlib_panel_alignment(fig,json_out=str(stem)+'.alignment.json',
        overlay_svg=str(stem)+'.alignment.svg',strict=True,tolerance_pt=1.5,gutter_tolerance_pt=1.5)
    fig.savefig(str(stem)+'.pdf')
    fig.savefig(str(stem)+'.svg')
    fig.savefig(str(stem)+'.png',dpi=600)
    if width==183:
        fig.savefig(str(stem)+'.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
        fig.savefig(output/'preview.png',dpi=300)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    output=args.output or next_experiment_path(ROOT/'outputs','mean_anchor_visual_architecture')
    output.mkdir(parents=True,exist_ok=True)
    data,files,manifest=load_data()
    draw(output,data,240);draw(output,data,183)
    record={'source_files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'input_indices':manifest['input_indices'],'full_volume_shapes':{k:list(data[k].shape) for k in ('mean','varvol','out')},
        'volume_display_limits_p999':data['limits'],'volume_gamma':.5,
        'volume_rendering':'orthogonal full-volume maximum projections mapped to three affine cube faces',
        'aspect_ratio':'schematic; cube dimensions are not calibrated physical lengths',
        'frame_thumbnails':'first 3 of the fixed 10 frames; all 10 underpin saved input statistics',
        'feature_planes':'abstract shapes, not learned activation measurements',
        'output_checkpoint':manifest['specified_old_checkpoint'],
        'output_example_role':'older Mean-anchor 400-step illustrative output, not current 800-step result',
        'historical_input_limitation':manifest['known_limitation'],
        'purpose':'architecture illustration, not reconstruction quality comparison',
        'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'image_provenance.json').write_text(json.dumps(record,indent=2,ensure_ascii=False))
    (output/Path(__file__).name).write_text(Path(__file__).read_text())
    print(output,flush=True)


if __name__=='__main__':main()
