"""P12 revision: separate irregular cell rings, morphology preview only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.io import savemat, loadmat
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import run_root_cell_preview as old


def cells(seed):
    rng=np.random.default_rng(seed)
    result=[]
    # Large cells are laid down first; many smaller cells fill their gaps.
    radii=np.concatenate((rng.uniform(9,12.5,24),rng.uniform(5,8.8,96),rng.uniform(2.8,5,130)))
    for radius in radii:
        aspect=rng.uniform(.72,1.22)
        a,b=radius*np.sqrt(aspect),radius/np.sqrt(aspect)
        accepted=False
        for trial in range(2400):
            upper=rng.random()<.49
            cx,cy=(143,86) if upper else (145,205)
            angle=rng.uniform(0,2*np.pi); rr=np.sqrt(rng.uniform(0,1))
            x=cx+rr*94*np.cos(angle); y=cy+rr*64*np.sin(angle)
            if rng.random()<.06:
                x,y=rng.uniform(93,198),rng.uniform(127,163)
            # Retain every complete ring including its wall support.
            margin=max(a,b)*1.20+3
            if min(x,y)<margin+10 or max(x,y)>280-margin:
                continue
            z=53+.065*(y-145)+3.5*np.sin(x/38)+rng.normal(0,3)
            # A minority lie at a distinct depth and may overlap in XY.
            secondary=rng.random()<.14
            if secondary:
                z+=rng.choice([-1,1])*13
            z=float(np.clip(z,29,78))
            if result:
                positions=np.array([c['center_xyz_um'] for c in result])
                distance=np.linalg.norm(positions[:,:2]-[x,y],axis=1)
                sums=radius+np.array([c['equivalent_radius_um'] for c in result])
                separate_z=np.abs(positions[:,2]-z)>12
                minimum=np.where(separate_z,.40,.95)*sums
                if np.any(distance<minimum):
                    continue
            accepted=True
            break
        if not accepted:
            continue
        result.append(dict(cell_id=len(result)+1,center_xyz_um=[float(x),float(y),z],
            equivalent_radius_um=float(radius),ellipse_radii_um=[float(a),float(b)],
            rotation_rad=float(rng.uniform(0,2*np.pi)),
            radial_harmonics=rng.uniform(.025,.070,3).tolist(),
            phases_rad=rng.uniform(0,2*np.pi,3).tolist(),
            wall_sigma_um=float(rng.uniform(.55,.85)),
            half_height_um=float(rng.uniform(1.8,3.2)),
            axial_taper_sigma_um=float(rng.uniform(.9,1.3)),
            tilt_xy=rng.uniform(-.18,.18,2).tolist(),
            amplitude=float(rng.uniform(.35,1.0)),
            angular_brightness_contrast=float(rng.uniform(.10,.32)),
            secondary_depth_proposal=bool(secondary)))
    assert len(result)>200
    return result


def render(items):
    axis=(np.arange(520)/2-.25)*old.PITCH
    zaxis=np.arange(5.5,105,1)
    fine_hi=np.zeros((520,520,100),np.float32)
    lumen_checks=[]
    for c in items:
        cx,cy,cz=c['center_xyz_um']; a,b=c['ellipse_radii_um']
        sigma=c['wall_sigma_um']; margin=max(a,b)*1.25+3*sigma
        ix=np.flatnonzero(np.abs(axis-cx)<=margin)
        iy=np.flatnonzero(np.abs(axis-cy)<=margin)
        max_z=margin*sum(np.abs(c['tilt_xy']))+c['half_height_um']+3*c['axial_taper_sigma_um']
        iz=np.flatnonzero(np.abs(zaxis-cz)<=max_z)
        x,y=np.meshgrid(axis[ix]-cx,axis[iy]-cy)
        phi=c['rotation_rad']; u=x*np.cos(phi)+y*np.sin(phi); v=-x*np.sin(phi)+y*np.cos(phi)
        t=np.arctan2(v/b,u/a)
        rho=np.sqrt((u/a)**2+(v/b)**2)
        contour=np.ones_like(t); derivative=np.zeros_like(t)
        for n,h,p in zip((3,4,5),c['radial_harmonics'],c['phases_rad']):
            contour+=h*np.cos(n*t+p)
            derivative-=n*h*np.sin(n*t+p)
        # First-order physical signed distance to the smooth implicit ring.
        safe_rho=np.maximum(rho,1e-6)
        drdu=u/(a*a*safe_rho); drdv=v/(b*b*safe_rho)
        dtdu=-v/(a*b*np.maximum(rho**2,1e-12)); dtdv=u/(a*b*np.maximum(rho**2,1e-12))
        gradient=np.sqrt((drdu-derivative*dtdu)**2+(drdv-derivative*dtdv)**2)
        distance=np.abs(rho-contour)/np.maximum(gradient,1/max(a,b))
        # Explicitly retain the dark lumen: distance approximation only near wall.
        valid=(rho>.55)&(rho<1.5)&(distance<3*sigma)
        angular=1+c['angular_brightness_contrast']*np.cos(2*t+c['phases_rad'][0])
        angular+=.10*np.sin(5*t+c['phases_rad'][1])
        wall=c['amplitude']*angular*np.exp(-.5*(distance/sigma)**2)*valid
        local_center=cz+c['tilt_xy'][0]*x+c['tilt_xy'][1]*y
        dz=np.maximum(np.abs(zaxis[iz][None,None,:]-local_center[:,:,None])-c['half_height_um'],0)
        sleeve=np.exp(-.5*(dz/c['axial_taper_sigma_um'])**2)
        sleeve[dz>3*c['axial_taper_sigma_um']]=0
        volume=(wall[:,:,None]*sleeve).astype(np.float32)
        target=np.ix_(iy,ix,iz)
        fine_hi[target]=np.maximum(fine_hi[target],volume)
        lumen_checks.append(float(wall[rho<.35].max(initial=0)))
    assert max(lumen_checks)==0
    fine=fine_hi.reshape(260,2,260,2,100).mean(axis=(1,3),dtype=np.float32)
    fine/=fine.max()
    coarse=fine.reshape(260,260,10,10).mean(axis=3,dtype=np.float32)
    return fine,coarse


def compare(previous, coarse, out):
    import matplotlib.pyplot as plt
    original=np.load(previous/'P12/ground_truth_YXZ.npy')
    fig,axes=plt.subplots(1,2,figsize=(13,7),layout='constrained')
    for ax,vol,title in zip(axes,(original,coarse),('Previous: connected polygon walls','Revised: separate irregular rings')):
        ax.imshow(vol.max(axis=2),cmap='inferno',vmin=0,vmax=vol.max(),interpolation='nearest')
        ax.set_title(title); ax.axis('off')
    fig.suptitle('P12 morphology revision | actual ground truth | per-volume global display scale')
    fig.savefig(out,dpi=160); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=2026090902)
    args=p.parse_args()
    previous=old.REPO/'data/root_cell_P12_preview_20260909_run01'
    protected_paths=[previous/'P12/truth.mat',old.SOURCE/'dataset_splits.json',old.SOURCE/'FINAL_DATASET_MANIFEST.json']
    protected={str(q):old.digest(q) for q in protected_paths}
    root=old.next_experiment_path(old.REPO/'data','root_cell_P12_rings_preview')
    root.mkdir(); folder=root/'P12'; folder.mkdir(); preview=folder/'previews'; preview.mkdir()
    items=cells(args.seed); fine,coarse=render(items)
    prior=loadmat(previous/'P12/truth.mat',simplify_cells=True)
    cfg=prior['cfg'].copy()
    cfg.update(geometry_version='root_irregular_rings_v2',geometry_kind='independent_irregular_ring_sleeves',
               description='Irregular independent rings with mixed sizes and depths',geometry_seed=args.seed,
               output_root=str(root),sample_dir=str(folder),stage='truth_preview_only',morphology_approved=False)
    for key in ('wall_sigma_um','wall_nominal_fwhm_um','wall_support_radius_um','axial_taper_sigma_um'):
        cfg.pop(key,None)
    cfg['wall_sigma_range_um']=[.55,.85]
    cfg['cell_count']=len(items)
    inputs,holdout=prior['input_indices'],prior['holdout_indices']
    metrics=old.validate(fine,coarse,inputs,holdout)
    radii=np.array([c['equivalent_radius_um'] for c in items])
    centers=np.array([c['center_xyz_um'] for c in items])
    dxy=np.linalg.norm(centers[:,None,:2]-centers[None,:,:2],axis=2)
    overlap=(dxy<radii[:,None]+radii[None,:])&np.triu(np.ones(dxy.shape,dtype=bool),1)
    metrics.update(cell_count=len(items),nominal_equivalent_diameter_um_range=[float(2*radii.min()),float(2*radii.max())],
                   nominal_equivalent_diameter_um_percentiles=np.percentile(2*radii,[10,50,90]).tolist(),
                   center_z_range_um=[float(centers[:,2].min()),float(centers[:,2].max())],
                   approximate_circle_projected_overlap_pairs=int(overlap.sum()),
                   isolated_cell_lumen_density_zero=True)
    source_hash=old.digest(__file__)
    meta=dict(array_axis_order='YXZ',geometry_axis_order='XYZ',geometry_units='um',
              normalization='one global fine-volume peak followed by slab average',
              coarse_voxel_semantics='average density in centered 10 um slabs',
              fine_grid_semantics='1 um Z quadrature; native XY integrated with 2x2 subpixels',
              morphology='independent smoothly deformed elliptical ring sleeves; dark lumens; no fluorescent end caps',
              compositing='maximum union; no additive intersection brightening',
              biological_claim='reference-inspired synthetic morphology, not recovered anatomy',
              geometry=np.array(items,dtype=object),reference_image_sha256=old.digest(old.REFERENCE),
              generator_dependencies_sha256=json.dumps({str(Path(__file__).resolve()):source_hash,str(Path(old.__file__).resolve()):old.digest(old.__file__)}))
    savemat(folder/'truth.mat',dict(cfg=cfg,meta=meta,ground_truth=coarse,ground_truth_fine=fine,
            metrics=metrics,source_sha256=source_hash,approval_status='pending_user_morphology_review',
            input_indices=inputs,holdout_indices=holdout),long_field_names=True,do_compression=True)
    roundtrip=loadmat(folder/'truth.mat')
    assert np.array_equal(roundtrip['ground_truth'],coarse) and np.array_equal(roundtrip['ground_truth_fine'],fine)
    np.save(folder/'ground_truth_YXZ.npy',coarse)
    for name,array in [('ground_truth_float.tif',coarse),('ground_truth_fine_float.tif',fine)]:
        tifffile.imwrite(folder/name,array.transpose(2,0,1),metadata={'axes':'ZYX'},photometric='minisblack')
    old.write_json(folder/'geometry.json',dict(version=2,cells=items,source_sha256=source_hash))
    old.write_json(folder/'truth_metrics.json',metrics)
    old.plots(preview,fine,coarse,centers[:,:2])
    compare(previous,coarse,preview/'P12_before_after.png')
    (root/'source_snapshot').mkdir()
    for source in (Path(__file__),Path(old.__file__)):
        (root/'source_snapshot'/source.name).write_bytes(source.read_bytes())
    assert protected=={str(q):old.digest(q) for q in protected_paths}
    old.write_json(root/'MORPHOLOGY_REVIEW_REQUIRED.json',dict(stage='truth_preview_only',morphology_approved=False,
           simulation_started=False,training_started=False,intended_sample_id='P12',intended_split='train',
           allowed_physical_gpu_indices=[0],no_gpu_fallback=True,preview_compute='CPU_ONLY',
           protected_source_hashes=protected,truth_sha256=old.digest(folder/'truth.mat'),metrics=metrics))
    old.write_json(root/'dataset_extension_draft.json',dict(dataset_complete=False,active_training_manifest_modified=False,
           proposed_addition=dict(sample_id='P12',split='train',sample_dir=str(folder),stage='truth_preview_only')))
    (root/'REPORT_ZH.md').write_text(f'''# P12 第二版：大小不一的不规则环形细胞

根据反馈，改为独立的闭合环形细胞，错落排列在上下两个疏密不均的区域。环有椭圆、拉长、轻微凹凸等变化，沿环亮度也不均匀；部分细胞在不同深度形成投影重叠。

- 环形细胞：{len(items)} 个。
- 名义等效直径：{2*radii.min():.1f}–{2*radii.max():.1f} µm；10/50/90百分位：{np.percentile(2*radii,[10,50,90]).round(1).tolist()} µm。轮廓扰动后实际尺寸会略有变化。
- 壁的高斯σ：0.55–0.85 µm，名义半高宽约1.3–2.0 µm。环内部为空，各环均有有限轴向厚度，细胞中心深度 {centers[:,2].min():.1f}–{centers[:,2].max():.1f} µm。
- 重叠处取最大密度，避免简单相加产生过亮交叉点。较深细胞可能投影到另一细胞的腔内。
- 真值尺寸和原数据一致：260×260×10，Z=10–100 µm；细网格Z采样1 µm，XY像素2×2子采样积分。
- 所有原生层共用同一亮度尺度；没有对图像施加模糊或制造相机噪声。
- 当前仅为几何预览，原版真值和正式训练集保留。完整光学仿真需本版形态确认，固定物理0号GPU。

[前后对比](P12/previews/P12_before_after.png) · [新版俯视图](P12/previews/P12_xy.png) · [深度预览](P12/previews/P12_overview.png) · [原生十层](P12/previews/P12_native_layers.png)

它是参考形态构造的合成样本；环的尺寸、深度和亮度不是从照片标定得到的真实菠菜根参数。用于训练前仍需完成P12新几何类型的数据适配与完整仿真验收。
''')
    old.write_json(root/'preview_artifacts_sha256.json',{str(q.relative_to(root)):old.digest(q) for q in root.rglob('*') if q.is_file()})
    print(json.dumps(dict(output=str(root),metrics=metrics),ensure_ascii=False))


if __name__=='__main__':
    main()
