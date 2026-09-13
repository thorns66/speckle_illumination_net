"""Native-pixel measurements and geometry-only P12 preview; no acquisition entry."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('MPLCONFIGDIR', '/tmp/p12_pixel_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat
from scipy.ndimage import map_coordinates, gaussian_filter1d
from scipy.signal import find_peaks, peak_widths
import tifffile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools import run_root_cell_preview as base
from tools import run_root_thin_preview as thin

REAL = REPO / 'data/菠菜根'
REAL_RUN = REPO / 'outputs/spinach_root_exploratory_20260909_run01'
PREVIOUS = REPO / 'data/root_cell_P12_thin_preview_20260909_run01/P12'
# Visually identified lumen centers and approximate search radii, in native
# zero-based pixels. Approximate radii only bound the wall-peak search.
OBJECT_ROIS = [
    (520,229,14),(454,312,24),(526,286,17),(579,259,15),(623,272,22),
    (656,292,24),(604,313,21),(400,335,22),(429,375,19),(491,373,27),
    (555,387,32),(619,368,31),(696,400,28),(651,445,25),(489,440,20),
    (682,498,25),(443,525,23),(497,566,21),(440,584,28),(650,558,29),
    (570,646,24),(497,660,23),(581,697,27),(666,686,22),(752,624,17),
    (693,629,26),(790,279,15),(706,296,14),(425,182,18),(484,725,12),
]
SENSOR_ROIS = [
    (449,322,5),(469,331,7),(499,317,6),(508,332,8),
    (524,335,8),(540,331,9),(563,334,9),(577,330,9),
    (601,333,8),(614,330,9),(470,369,8),(508,369,8),
    (525,373,8),(577,368,9),(602,372,8),(615,366,8),
]


def quantiles(values):
    return dict(zip(('p10','median','p90'), np.percentile(values,[10,50,90]).tolist()))


def measure(image, rois, source, directory):
    cells, rays, profiles = [], [], []
    for cell_id,(cx,cy,radius) in enumerate(rois,1):
        diameters, widths, contrasts = [], [], []
        endpoints=[]
        for angle_deg in (0,45,90,135):
            theta=np.deg2rad(angle_deg); side_peaks=[]
            for side in (-1,1):
                r=np.arange(0,1.65*radius+.25,.25)
                # Average a 3-pixel-wide stripe perpendicular to the profile.
                offsets=np.array([-1,0,1]) if source!='sensor_microimages' else np.array([-.5,0,.5])
                x=cx+side*r[None,:]*np.cos(theta)-offsets[:,None]*np.sin(theta)
                y=cy+side*r[None,:]*np.sin(theta)+offsets[:,None]*np.cos(theta)
                intensity=map_coordinates(image,[y,x],order=1).mean(axis=0)
                intensity=gaussian_filter1d(intensity,2)
                noise_floor=float(np.median(intensity[r<.3*radius]))
                prominence=.08*max(float(np.ptp(intensity)),1e-20)
                peaks,props=find_peaks(intensity,prominence=prominence)
                inside=(r[peaks]>.50*radius)&(r[peaks]<1.50*radius)
                peaks=peaks[inside]; prom=props['prominences'][inside]
                if not len(peaks):
                    side_peaks.append(None);continue
                score=prom*np.exp(-.5*((r[peaks]-radius)/(.35*radius))**2)
                peak=int(peaks[np.argmax(score)])
                distance=float(r[peak]); width=float(peak_widths(intensity,[peak],rel_height=.5)[0][0]*.25)
                contrast=float((intensity[peak]-noise_floor)/max(abs(intensity[peak]),1e-20))
                if contrast<.12:
                    side_peaks.append(None);continue
                px=float(cx+side*distance*np.cos(theta)); py=float(cy+side*distance*np.sin(theta))
                side_peaks.append(distance);widths.append(width);contrasts.append(contrast);endpoints.append([px,py])
                rays.append(dict(source=source,cell_id=cell_id,center_x_px=cx,center_y_px=cy,
                    angle_deg=angle_deg,side=side,wall_x_px=px,wall_y_px=py,
                    radial_wall_distance_px=distance,apparent_wall_fwhm_px=width,contrast=contrast))
                profiles.append(dict(cell_id=cell_id,angle=angle_deg,side=side,r=r,
                    intensity=intensity,peak=peak))
            if all(p is not None for p in side_peaks): diameters.append(sum(side_peaks))
        accepted=len(diameters)>=3
        cells.append(dict(source=source,cell_id=cell_id,center_x_px=cx,center_y_px=cy,
            approximate_search_radius_px=radius,accepted=accepted,
            valid_axes=len(diameters),diameter_median_px=float(np.median(diameters)) if diameters else 0,
            diameter_min_px=min(diameters,default=0),diameter_max_px=max(diameters,default=0),
            apparent_wall_fwhm_px=float(np.median(widths)) if widths else 0,
            contrast_median=float(np.median(contrasts)) if contrasts else 0))
    for name,rows in [(source+'_cells.csv',cells),(source+'_rays.csv',rays)]:
        with (directory/name).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return cells,rays,profiles


def make_cells(diameters,seed=2026090904,count=18):
    rng=np.random.default_rng(seed)
    # Retain the measured 10-90% size range, avoiding uncertain extremes.
    target=np.quantile(diameters,np.linspace(.10,.90,count))
    target=np.sort(target)[::-1]
    items=[]
    for diameter in target:
        radius=diameter*base.PITCH/2+.60
        aspect=float(rng.uniform(.84,1.17));a=radius*np.sqrt(aspect);b=radius/np.sqrt(aspect)
        bound=1.12*max(a,b)+9
        for attempt in range(30000):
            x,y=rng.uniform(bound,259*base.PITCH-bound,2)
            if items:
                centers=np.array([c['center_xyz_um'][:2] for c in items])
                sums=radius+np.array([c['nominal_radius_um'] for c in items])
                if np.any(np.linalg.norm(centers-[x,y],axis=1)<.93*sums):continue
            break
        else:raise RuntimeError(f'Cannot place measured cell {len(items)+1}; no silent shrinking')
        items.append(dict(cell_id=len(items)+1,center_xyz_um=[float(x),float(y),float(thin.middle_z(x,y))],
            nominal_radius_um=float(radius),target_wall_peak_diameter_px=float(diameter),
            ellipse_radii_um=[float(a),float(b)],rotation_rad=float(rng.uniform(0,2*np.pi)),
            harmonics=rng.uniform(.012,.030,3).tolist(),phases=rng.uniform(0,2*np.pi,3).tolist(),
            wall_sigma_um=float(rng.uniform(.65,.90)),amplitude=float(rng.uniform(.55,1)),
            brightness_contrast=float(rng.uniform(.05,.14))))
    return items


def figures(root,raw,mean,taylor,cells,rays,profiles,coarse,old,synthetic):
    out=root/'P12/previews';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(15,10),layout='constrained')
    axes[0].imshow(raw,cmap='inferno',vmin=0,vmax=np.percentile(raw,99.8))
    axes[0].set(title='Original sensor | 1421 x 1029 native pixels',xlabel='Native X pixel',ylabel='Native Y pixel')
    axes[0].add_patch(plt.Rectangle((430,280),200,140,fill=False,ec='cyan',lw=1))
    axes[1].imshow(mean,cmap='inferno',vmin=0,vmax=np.percentile(mean,99.8))
    for cell in cells:
        c='cyan' if cell['accepted'] else 'red'
        axes[1].text(cell['center_x_px'],cell['center_y_px'],str(cell['cell_id']),color=c,fontsize=8,ha='center')
    for ray in rays:
        axes[1].plot(ray['wall_x_px'],ray['wall_y_px'],'.',ms=2,color='lime')
    axes[1].set(xlim=(350,950),ylim=(810,140),title='Mean RL3 at 50 um | measured wall peaks',xlabel='Native X pixel',ylabel='Native Y pixel')
    fig.savefig(out/'real_ring_measurements.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,9),layout='constrained')
    for ax,cellid in zip(axes.flat,[2,6,10,12,17,20]):
        for pr in profiles:
            if pr['cell_id']==cellid and pr['angle'] in (0,90):
                signal=pr['intensity']/max(pr['intensity'].max(),1e-20)
                ax.plot(pr['side']*pr['r'],signal,label=f"{pr['angle']}deg side {pr['side']}")
                ax.plot(pr['side']*pr['r'][pr['peak']],signal[pr['peak']],'o')
        ax.set(title=f"Cell {cellid}: {cells[cellid-1]['diameter_median_px']:.1f} px",xlabel='Distance from center (native px)',ylabel='Relative intensity');ax.grid(alpha=.25)
    axes.flat[0].legend(fontsize=7);fig.savefig(out/'measurement_profiles.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(16,4.8),layout='constrained')
    real_crop=mean[270:530,380:640]; taylor_crop=taylor[270:530,380:640]
    for ax,v,title in zip(axes,[real_crop,taylor_crop,old.max(2),coarse.max(2)],
        ['Real Mean RL3: native crop','Real Taylor RL3: same crop','Rejected: 225 small cells','Revised: 18 large cells']):
        ax.imshow(v,cmap='inferno',vmin=0,vmax=v.max(),interpolation='nearest',extent=[0,260,260,0]);ax.set(title=title,xlabel='Native pixels',ylabel='Native pixels');ax.set_xticks([0,65,130,195,260]);ax.set_yticks([0,65,130,195,260])
        ax.plot([15,65],[242,242],c='white',lw=3);ax.text(40,233,'50 px',color='white',ha='center')
    fig.suptitle('All panels cover 260 x 260 native pixels; no resizing of the full real field\nEach panel uses one global maximum; appearance only, not an intensity calibration')
    fig.savefig(out/'native_pixel_comparison.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,8),layout='constrained');ax.imshow(coarse.max(2),cmap='inferno',vmin=0,vmax=coarse.max(),interpolation='nearest',extent=[0,260,260,0]);ax.set(title='P12 revised GT | 260 x 260 | 18 cells',xlabel='X (native pixels)',ylabel='Y (native pixels)');ax.plot([15,65],[243,243],color='white',lw=3);ax.text(40,234,'50 pixels',color='white',ha='center');fig.savefig(out/'P12_xy_pixels.png',dpi=170);plt.close(fig)
    vals=[c['diameter_median_px'] for c in cells if c['accepted']]
    new=[c['diameter_median_px'] for c in synthetic if c['accepted']]
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained');ax.hist(vals,bins=np.arange(10,90,5),alpha=.5,label='Measured real Mean RL3 rings');ax.hist(new,bins=np.arange(10,90,5),alpha=.5,label='Synthetic GT rings');ax.set(xlabel='Median wall-peak diameter (native px)',ylabel='Count');ax.legend();fig.savefig(out/'diameter_distribution.png',dpi=150);plt.close(fig)


def main():
    protected={str(base.SOURCE/n):base.digest(base.SOURCE/n) for n in ['dataset_splits.json','FINAL_DATASET_MANIFEST.json','python_reader_verification.json']}
    root=base.next_experiment_path(REPO/'data','root_cell_P12_pixel_preview');root.mkdir()
    folder=root/'P12';folder.mkdir(); measurements=root/'measurements';measurements.mkdir()
    files=sorted(REAL.glob('50_*.tif'));assert len(files)==100
    manifest=json.loads((REAL_RUN/'manifest.json').read_text())
    for p in files:
        assert base.digest(p)==manifest['input_files_sha256'][str(p)]
    raw=np.zeros((1029,1421),np.float64)
    for p in files:raw+=tifffile.imread(p)/100
    volume_path=REAL_RUN/'volumes/mean_rl3/reconstruction.npy'
    taylor_path=REAL_RUN/'volumes/taylor_rl3_sqrt/reconstruction.npy'
    mean=np.load(volume_path,mmap_mode='r')[4]; taylor=np.load(taylor_path,mmap_mode='r')[4]
    assert mean.shape==raw.shape==taylor.shape
    cells,rays,profiles=measure(mean,OBJECT_ROIS,'object_mean_rl3',measurements)
    micro,_,_=measure(raw,SENSOR_ROIS,'sensor_microimages',measurements)
    companion,_,_=measure(taylor,OBJECT_ROIS,'object_taylor_rl3',measurements)
    diameters=np.array([c['diameter_median_px'] for c in cells if c['accepted']])
    assert len(diameters)>=24
    items=make_cells(diameters)
    fine,coarse,labels,wall_labels,audit=thin.rasterize(items)
    synthetic_rois=[(c['center_xyz_um'][0]/base.PITCH,c['center_xyz_um'][1]/base.PITCH,
        c['target_wall_peak_diameter_px']/2) for c in items]
    synthetic,_,_=measure(coarse[:,:,4],synthetic_rois,'synthetic_gt',measurements)
    prior=loadmat(PREVIOUS/'truth.mat',simplify_cells=True);cfg=prior['cfg'].copy()
    cfg.update(geometry_version='root_native_pixel_v4',geometry_kind='nonoverlapping_deformed_cell_section',
        description='18 large rings calibrated to native-pixel real Mean-RL3 wall diameters',
        geometry_seed=2026090904,output_root=str(root),sample_dir=str(folder),cell_count=len(items),
        morphology_approved=False,stage='truth_preview_only',wall_sigma_range_um=[.65,.90],
        size_calibration='native real Mean-RL3 wall-peak diameters; no full-field resize',
        physical_size_calibration='simulation pitch only; not a measured real micrometre calibration')
    # MATLAB-native numeric config types; density arrays remain float32.
    for name,value in list(cfg.items()):
        if isinstance(value,np.ndarray) and np.issubdtype(value.dtype,np.integer):cfg[name]=value.astype(np.float64)
        elif isinstance(value,(int,np.integer)) and not isinstance(value,bool):cfg[name]=float(value)
    inputs=np.array(prior['input_indices'],dtype=np.float64);holdout=np.array(prior['holdout_indices'],dtype=np.float64)
    metrics=base.validate(fine,coarse,inputs,holdout)
    metrics.update({k:v for k,v in audit.items() if k not in ('cells','contact_pairs_one_based')})
    calibration=dict(native_raw_shape_yx=list(raw.shape),real_resizing_applied=False,
        selected_real_object_ring_count=len(cells),accepted_real_object_ring_count=len(diameters),
        real_object_diameter_px=quantiles(diameters),real_sensor_microimage_diameter_px=quantiles([c['diameter_median_px'] for c in micro if c['accepted']]),
        synthetic_wall_peak_diameter_px=quantiles([c['diameter_median_px'] for c in synthetic if c['accepted']]),
        previous_filled_cell_diameter_px=quantiles([2*np.sqrt(c['filled_area_um2']/np.pi)/base.PITCH for c in json.loads((PREVIOUS/'geometry.json').read_text())['nonoverlap']['cells']]),
        apparent_real_wall_fwhm_px=quantiles([c['apparent_wall_fwhm_px'] for c in cells if c['accepted']]),
        measurement_definition='Median of >=3 of 4 opposed wall-peak distances (0,45,90,135 deg) at fixed visually selected lumen centers',
        selection_limit='30 clearly recognizable rings; not an unbiased survey; same-source overlapping views and processed reconstructions are not independent biological repeats',
        sensor_microimages_limit='Light-field subimages may show repeated views of the same object; their pixel diameter is not object-grid cell diameter',
        density_wall_note='Apparent blurred wall widths were not copied into GT. GT sigma=.65-.90 um on simulation pitch; optical blur not baked in',
        biological_calibration='No known specimen micrometre calibration; TIFF 72 dpi is not microscopy pixel pitch',
        no_forward_or_network_run=True)
    metrics['calibration']=calibration
    sha=base.digest(__file__)
    meta=dict(array_axis_order='YXZ',geometry_axis_order='XYZ',geometry_units='um',
        geometry=np.array(items,dtype=object),normalization='one global fine-volume peak then 10 um slab means',
        calibration=calibration,biological_claim='size-guided synthetic thin tissue patch, not measured biological GT')
    savemat(folder/'truth.mat',dict(cfg=cfg,meta=meta,ground_truth=coarse,ground_truth_fine=fine,
        metrics=metrics,source_sha256=sha,input_indices=inputs,holdout_indices=holdout,
        approval_status='pending_user_review_large_native_pixel_rings'),do_compression=True,long_field_names=True)
    np.save(folder/'ground_truth_YXZ.npy',coarse);np.save(folder/'cell_region_labels_YX_3x.npy',labels);np.save(folder/'wall_region_labels_YX_3x.npy',wall_labels)
    for name,v in [('ground_truth_float.tif',coarse),('ground_truth_fine_float.tif',fine)]:
        tifffile.imwrite(folder/name,v.transpose(2,0,1),metadata={'axes':'ZYX'},photometric='minisblack')
    base.write_json(folder/'geometry.json',dict(cells=items,nonoverlap=audit,source_sha256=sha))
    base.write_json(folder/'truth_metrics.json',metrics);base.write_json(root/'calibration.json',calibration)
    (folder/'previews').mkdir();base.plots(folder/'previews',fine,coarse,np.array([c['center_xyz_um'][:2] for c in items]))
    figures(root,raw,mean,taylor,cells,rays,profiles,coarse,prior['ground_truth'],synthetic)
    source_record=dict(raw_files_sha256={str(p):base.digest(p) for p in files},
        real_run_manifest_sha256=base.digest(REAL_RUN/'manifest.json'),
        measurement_mean_volume_sha256=base.digest(volume_path),crosscheck_taylor_volume_sha256=base.digest(taylor_path),
        previous_truth_sha256=base.digest(PREVIOUS/'truth.mat'),code_sha256={str(p):base.digest(p) for p in [Path(__file__),Path(thin.__file__),Path(base.__file__)]})
    base.write_json(root/'source_provenance.json',source_record)
    snapshot=root/'source_snapshot';snapshot.mkdir()
    for p in [Path(__file__),Path(thin.__file__),Path(base.__file__)]:shutil.copy2(p,snapshot/p.name)
    assert not (base.SOURCE/'P12').exists()
    assert protected=={p:base.digest(p) for p in protected}
    base.write_json(root/'MORPHOLOGY_REVIEW_REQUIRED.json',dict(morphology_approved=False,simulation_started=False,
        training_started=False,intended_sample_id='P12',intended_split='train',allowed_physical_gpu_indices=[0],
        truth_sha256=base.digest(folder/'truth.mat'),protected_dataset_manifests=protected))
    d=calibration['real_object_diameter_px'];n=calibration['synthetic_wall_peak_diameter_px'];s=calibration['real_sensor_microimage_diameter_px'];old=calibration['previous_filled_cell_diameter_px']
    (root/'REPORT_ZH.md').write_text(f'''# P12 大圆环真值预览（待确认）

已停止并删除总数据集内被否决的 P12。正式数据集仍为 110/30/30 个子集，原指纹恢复。本目录只保存测量和真值，没有生成传感器帧或开始训练。

## 原始数据实际测量

依据 data/菠菜根 中 100 张 1421×1029 原生 TIFF。没有把整幅图缩成 260×260。原始 sensor 是光场记录，一个细胞可产生多视角小像；不能将小像当作重建体中的独立细胞。

- 原生 sensor 中选取 16 个清楚小像，测得壁峰间距中位数 {s['median']:.1f} px，10–90% 范围 {s['p10']:.1f}–{s['p90']:.1f} px。这些小像可能是同一细胞的重复视角，不是 16 个独立细胞。
- 对同一批原始数据现有 Mean-RL3 的 50 µm 原生层，人工标定 30 处可辨认细胞腔中心；四个方向各测两侧亮壁峰间距，至少三轴成功才保留。保留 {len(diameters)} 处，中位直径 {d['median']:.1f} px，10–90% 范围 {d['p10']:.1f}–{d['p90']:.1f} px。Taylor 同位置结果仅作交叉核对，网络输出未用于定标。
- 上版 225 颗细胞外轮廓等效直径中位数 {old['median']:.1f} px，明显小于真实重建图中的细胞。这是比较不同但接近的几何直径定义，不宣称精确放大倍率。
- 每处中心、四轴壁峰坐标、失败轴和直径均有 CSV；这是对清楚细胞的代表性测量，不是对全部真实细胞的无偏统计，更不是真实密度 GT。

## 新真值

保持 260×260×10，改成 {len(items)} 个大小不一的大环；对新 GT 用同一规则测量，壁峰间距中位数 {n['median']:.1f} px，10–90% 范围 {n['p10']:.1f}–{n['p90']:.1f} px。从真实大图取 260×260 原生裁块作对照，模拟的是局部组织，而非把整片根压进一个小视野。

细胞互不穿插，接触处互补形变；细胞腔保持空，225 个小环减少为 {len(items)} 个大环。仅 40/50/60 µm 三层非零，局部厚度 14 µm。细胞壁宽度未照搬经过光学模糊的实测宽度，避免把模糊预先写进 GT。像素对应的 µm 仅沿用仿真网格，72 dpi TIFF 标签不能用于推算真实物理尺寸。

## 预览文件

- P12/previews/native_pixel_comparison.png：真实 Mean/Taylor 同位置 260×260 裁块、上版、新版。
- P12/previews/P12_xy_pixels.png：新版 XY 真值与 50 像素标尺。
- P12/previews/P12_native_layers.png：十层统一亮度。
- P12/previews/real_ring_measurements.png：原图与测量标注。
- P12/previews/measurement_profiles.png：固定中心实际剖面和取峰位置。

新预览仍需用户确认。未进行任何新前向模拟，因此当前不能宣称新 sensor 已清晰；正式仿真仍按用户要求使用 0 号 GPU。
''',encoding='utf-8')
    base.write_json(root/'preview_artifacts_sha256.json',{str(p.relative_to(root)):base.digest(p) for p in root.rglob('*') if p.is_file()})
    print(json.dumps(dict(output=str(root),calibration=calibration,cell_count=len(items),touching_pairs=audit['touching_pairs']),ensure_ascii=False))


if __name__=='__main__':main()
