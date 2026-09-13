"""P12 v3: a thin, non-overlapping cellular section; no acquisition entry point."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage import distance_transform_edt, gaussian_filter, label, binary_fill_holes
import tifffile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import run_root_cell_preview as base

SS=3
STEP=base.PITCH/SS
SHAPE=(260*SS,260*SS)


def middle_z(x,y):
    return 50+1.5*np.sin((x-145)/69)+np.cos((y-145)/72)


def make_cells(seed):
    rng=np.random.default_rng(seed)
    radii=np.concatenate([rng.uniform(9,12,20),rng.uniform(5.4,8.6,90),rng.uniform(3.2,5.2,115)])
    items=[]
    for radius in radii:
        aspect=rng.uniform(.83,1.18)
        a,b=radius*np.sqrt(aspect),radius/np.sqrt(aspect)
        accepted=False
        for _ in range(4500):
            cx,cy=(144,87) if rng.random()<.49 else (145,204)
            t=rng.uniform(0,2*np.pi); rr=np.sqrt(rng.uniform(0,1))
            x,y=cx+94*rr*np.cos(t),cy+64*rr*np.sin(t)
            if rng.random()<.045:
                x,y=rng.uniform(105,188),rng.uniform(137,155)
            margin=max(a,b)*1.15+3
            if min(x,y)<margin+12 or max(x,y)>279-margin:
                continue
            if items:
                centers=np.array([c['center_xyz_um'][:2] for c in items])
                sums=radius+np.array([c['nominal_radius_um'] for c in items])
                if np.any(np.linalg.norm(centers-[x,y],axis=1)<.92*sums):
                    continue
            accepted=True
            break
        if not accepted:
            continue
        items.append(dict(cell_id=len(items)+1,center_xyz_um=[float(x),float(y),float(middle_z(x,y))],
             nominal_radius_um=float(radius),ellipse_radii_um=[float(a),float(b)],
             rotation_rad=float(rng.uniform(0,2*np.pi)),
             harmonics=rng.uniform(.012,.035,3).tolist(),phases=rng.uniform(0,2*np.pi,3).tolist(),
             wall_sigma_um=float(rng.uniform(.50,.65)),amplitude=float(rng.uniform(.50,1)),
             brightness_contrast=float(rng.uniform(.05,.18))))
    assert len(items)>=190
    return items


def rasterize(items):
    axis=(np.arange(260*SS)/SS-(SS-1)/(2*SS))*base.PITCH
    winner=np.zeros(SHAPE,np.uint16)
    best=np.full(SHAPE,np.inf,np.float32)
    footprints=[]
    for c in items:
        cx,cy,_=c['center_xyz_um']; a,b=c['ellipse_radii_um']
        pad=max(a,b)*1.18+3
        ix=np.flatnonzero(abs(axis-cx)<pad); iy=np.flatnonzero(abs(axis-cy)<pad)
        x,y=np.meshgrid(axis[ix]-cx,axis[iy]-cy)
        angle=c['rotation_rad']
        u=x*np.cos(angle)+y*np.sin(angle); v=-x*np.sin(angle)+y*np.cos(angle)
        t=np.arctan2(v/b,u/a); rho=np.sqrt((u/a)**2+(v/b)**2)
        contour=np.ones_like(rho)
        for n,h,p in zip((3,4,5),c['harmonics'],c['phases']):
            contour+=h*np.cos(n*t+p)
        inside=rho<=contour
        # Smooth oval boundary competes with neighbours only where they contact.
        score=(rho-contour)*c['nominal_radius_um']
        dst=np.ix_(iy,ix)
        replace=inside&(score<best[dst])
        old_best=best[dst]; old_best[replace]=score[replace]; best[dst]=old_best
        old_winner=winner[dst]; old_winner[replace]=c['cell_id']; winner[dst]=old_winner
        footprints.append((iy,ix,inside))
    filled_labels=np.zeros_like(winner)
    wall_labels=np.zeros_like(winner)
    density=np.zeros(SHAPE,np.float32)
    overlap_count=np.zeros(SHAPE,np.uint8)
    summaries=[]
    for c,(iy,ix,ideal) in zip(items,footprints):
        dst=np.ix_(iy,ix); mask=winner[dst]==c['cell_id']
        # Rounding is applied to geometry, then restricted to the exclusive cell domain.
        smooth=gaussian_filter(mask.astype(np.float32),sigma=.40/STEP)
        mask &= smooth>=.46
        assert label(mask)[1]==1
        assert np.array_equal(binary_fill_holes(mask),mask)
        inside_distance=np.maximum(0,distance_transform_edt(mask,sampling=STEP)-.5*STEP)
        sigma=c['wall_sigma_um']
        # Entire fluorescence support stays inside its own filled cell region.
        # At contact, opposite halves of neighbouring walls can meet but never cross.
        wall_support=mask&(inside_distance<=.60+3*sigma)
        lumen=mask&~wall_support
        assert label(lumen)[1]==1 and lumen.sum()>3
        x,y=np.meshgrid(axis[ix]-c['center_xyz_um'][0],axis[iy]-c['center_xyz_um'][1])
        t=np.arctan2(y,x)
        brightness=c['amplitude']*(1+c['brightness_contrast']*np.cos(2*t+c['phases'][0]))
        wall=(brightness*np.exp(-.5*((inside_distance-.60)/sigma)**2)*wall_support).astype(np.float32)
        n=overlap_count[dst]; n+=mask; overlap_count[dst]=n
        cell=filled_labels[dst]; cell[mask]=c['cell_id']; filled_labels[dst]=cell
        w=wall_labels[dst]; w[wall_support]=c['cell_id']; wall_labels[dst]=w
        d=density[dst]; d[mask]=wall[mask]; density[dst]=d
        retained=float(mask.sum()/ideal.sum())
        summaries.append(dict(cell_id=c['cell_id'],filled_area_um2=float(mask.sum()*STEP**2),
            lumen_area_um2=float(lumen.sum()*STEP**2),retained_area_fraction=retained,
            filled_connected_components=1,lumen_connected_components=1,
            lumen_density_max=float(wall[lumen].max(initial=0))))
    assert overlap_count.max()==1 and np.count_nonzero(overlap_count>1)==0
    # Labels are fixed across Z; no cell may sit above another cell's XY region.
    # This stronger condition also forbids projected overlap before voxel averaging.
    xx,yy=np.meshgrid(axis,axis)
    center=middle_z(xx,yy)
    fine=np.zeros((260,260,100),np.float32)
    for k,z in enumerate(np.arange(5.5,105,1)):
        dz=abs(z-center)
        profile=np.where(dz<=5,1,np.where(dz<7,.5*(1+np.cos(np.pi*(dz-5)/2)),0))
        plane=(density*profile).astype(np.float32)
        fine[:,:,k]=plane.reshape(260,SS,260,SS).mean(axis=(1,3),dtype=np.float32)
    fine/=fine.max()
    coarse=fine.reshape(260,260,10,10).mean(axis=3,dtype=np.float32)
    assert np.count_nonzero(coarse[:,:,[0,1,2,6,7,8,9]])==0
    assert np.all(coarse.sum(axis=(0,1))[3:6]>0)
    # Sample at 3x native grid and report the resolution of this nonintersection audit.
    contact_pairs=set()
    for left,right in ((filled_labels[:,:-1],filled_labels[:,1:]),(filled_labels[:-1],filled_labels[1:])):
        touch=(left>0)&(right>0)&(left!=right)
        contact_pairs.update(tuple(sorted((int(a),int(b)))) for a,b in zip(left[touch],right[touch]))
    audit=dict(cell_count=len(items),geometry_sampling_pitch_um=STEP,
        filled_cell_overlap_subpixels=int(np.count_nonzero(overlap_count>1)),
        projected_cell_overlap_subpixels=int(np.count_nonzero(overlap_count>1)),
        maximum_cell_ownership=int(overlap_count.max()),
        all_cells_and_lumens_connected=True,all_individual_lumens_dark=True,
        touching_pairs=len(contact_pairs),contact_pairs_one_based=sorted(contact_pairs),
        contacts_are_complementary_boundaries=True,
        cell_region_fixed_in_xy_across_depth=True,
        local_slice_full_support_thickness_um=14,local_slice_plateau_thickness_um=10,
        requested_coarse_support_z_um=[40,50,60],out_of_allowed_layers_nonzero_voxels=0,
        equivalent_cell_diameter_um_percentiles_10_50_90=np.percentile(
            [2*np.sqrt(s['filled_area_um2']/np.pi) for s in summaries],[10,50,90]).tolist(),
        cells=summaries)
    return fine,coarse,filled_labels,wall_labels,audit


def extra_figures(folder,coarse,labels,previous):
    import matplotlib.pyplot as plt
    extent=[0,260*base.PITCH,260*base.PITCH,0]
    fig,axes=plt.subplots(1,2,figsize=(13,7),layout='constrained')
    for ax,vol,title in zip(axes,[previous,coarse],['Previous: overlapping thick specimen','Revised: contact deformation, thin section']):
        ax.imshow(vol.max(axis=2),cmap='inferno',vmin=0,vmax=vol.max(),extent=extent,interpolation='nearest')
        ax.set_title(title); ax.axis('off')
    fig.savefig(folder/'P12_before_after.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,5.5),layout='constrained')
    axes[0].imshow(coarse.max(axis=2),cmap='inferno',vmin=0,vmax=coarse.max(),extent=extent,interpolation='nearest')
    for ax in axes:
        ax.set(xlim=(100,178),ylim=(120,50),xlabel='X (um)',ylabel='Y (um)')
    axes[0].set_title('Contact detail | native GT')
    color=np.random.default_rng(42).random((int(labels.max())+1,3))*.7+.3;color[0]=0
    axes[1].imshow(color[labels],extent=extent,interpolation='nearest')
    axes[1].set_title('Filled cell regions | unique cell per subpixel')
    fig.savefig(folder/'P12_contact_audit.png',dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=2026090903)
    args=p.parse_args()
    before=base.REPO/'data/root_cell_P12_rings_preview_20260909_run01'
    protected_paths=[before/'P12/truth.mat',base.SOURCE/'dataset_splits.json',base.SOURCE/'FINAL_DATASET_MANIFEST.json']
    protected={str(q):base.digest(q) for q in protected_paths}
    root=base.next_experiment_path(base.REPO/'data','root_cell_P12_thin_preview')
    root.mkdir();folder=root/'P12';folder.mkdir();preview=folder/'previews';preview.mkdir()
    items=make_cells(args.seed);fine,coarse,labels,wall_labels,audit=rasterize(items)
    prior=loadmat(before/'P12/truth.mat',simplify_cells=True);cfg=prior['cfg'].copy()
    cfg.update(geometry_version='root_contact_thin_v3',geometry_kind='nonoverlapping_deformed_cell_section',
        description='Thin cellular section with rounded contact deformation and no projected overlap',
        geometry_seed=args.seed,output_root=str(root),sample_dir=str(folder),morphology_approved=False,
        stage='truth_preview_only',cell_count=len(items),wall_sigma_range_um=[.5,.65],
        lateral_quadrature_samples_per_pixel=SS,allowed_truth_z_um=[40,50,60],
        local_slice_full_support_thickness_um=14)
    inputs,holdout=prior['input_indices'],prior['holdout_indices']
    metrics=base.validate(fine,coarse,inputs,holdout)
    metrics.update({k:v for k,v in audit.items() if k not in ('cells','contact_pairs_one_based')})
    sha=base.digest(__file__)
    meta=dict(array_axis_order='YXZ',geometry_axis_order='XYZ',geometry_units='um',
        normalization='one global fine-volume peak followed by 10 um slab average',
        coarse_voxel_semantics='average density in centered 10 um slabs',
        fine_grid_semantics='1 um axial quadrature; 3x3 lateral quadrature per native pixel',
        morphology='rounded cell envelopes, compressed where neighbours contact, no interpenetration',
        fluorescence='finite inward wall band around a dark lumen; no fluorescent caps',
        packing='exclusive XY cell ownership at 3x native resolution; reused at every depth',
        biological_claim='user-reference-inspired thin section; exact anatomy and micrometre dimensions are not photo-calibrated',
        geometry=np.array(items,dtype=object),reference_image_sha256=base.digest(base.REFERENCE),
        plotting_dependency_sha256=base.digest(base.__file__))
    savemat(folder/'truth.mat',dict(cfg=cfg,meta=meta,ground_truth=coarse,ground_truth_fine=fine,
        metrics=metrics,source_sha256=sha,approval_status='pending_user_morphology_review',
        input_indices=inputs,holdout_indices=holdout),do_compression=True,long_field_names=True)
    loaded=loadmat(folder/'truth.mat');assert np.array_equal(loaded['ground_truth'],coarse)
    assert np.array_equal(loaded['ground_truth_fine'],fine)
    np.save(folder/'ground_truth_YXZ.npy',coarse)
    np.save(folder/'cell_region_labels_YX_3x.npy',labels)
    np.save(folder/'wall_region_labels_YX_3x.npy',wall_labels)
    for name,a in [('ground_truth_float.tif',coarse),('ground_truth_fine_float.tif',fine)]:
        tifffile.imwrite(folder/name,a.transpose(2,0,1),metadata={'axes':'ZYX'},photometric='minisblack')
    base.write_json(folder/'geometry.json',dict(cells=items,nonoverlap=audit,source_sha256=sha))
    base.write_json(folder/'truth_metrics.json',metrics)
    base.plots(preview,fine,coarse,np.array([c['center_xyz_um'][:2] for c in items]))
    extra_figures(preview,coarse,labels,prior['ground_truth'])
    (root/'source_snapshot').mkdir()
    for q in (Path(__file__),Path(base.__file__)):
        (root/'source_snapshot'/q.name).write_bytes(q.read_bytes())
    assert protected=={str(q):base.digest(q) for q in protected_paths}
    base.write_json(root/'MORPHOLOGY_REVIEW_REQUIRED.json',dict(stage='truth_preview_only',
        morphology_approved=False,simulation_started=False,training_started=False,
        intended_sample_id='P12',intended_split='train',allowed_physical_gpu_indices=[0],
        no_gpu_fallback=True,preview_compute='CPU_ONLY',truth_sha256=base.digest(folder/'truth.mat'),
        protected_source_hashes=protected,metrics=metrics))
    base.write_json(root/'dataset_extension_draft.json',dict(dataset_complete=False,
        active_training_manifest_modified=False,proposed_addition=dict(sample_id='P12',split='train',
        sample_dir=str(folder),stage='truth_preview_only')))
    sources=[
        dict(title='Taxonomic characteristics of Spinacia oleracea L. (Wahua and Agogbua, 2023), Plate 3d',
             url='https://www.ajol.info/index.php/sa/article/download/263438/248674',
             scope='Spinach-specific root anatomy: central xylem and surrounding tissue; searchable text available, publisher access blocked; no image measurement performed.'),
        dict(title='Kitin et al. (2020), Direct fluorescence imaging of lignocellulosic and suberized cell walls in roots and stems',
             url='https://www.fpl.fs.usda.gov/documnts/pdf2020/fpl_2020_kitin001.pdf',
             scope='Primary microscopy reference for plant cell walls, contacting envelopes and intercellular spaces; examples include mangrove, NOT spinach.'),
        dict(title='Schiefelbein Lab: Rapid preparation of transverse sections of plant roots',
             url='https://sites.lsa.umich.edu/schiefelbein-lab/rapid-preparation-of-transverse-sections-of-plant-roots/',
             scope='Preparation, fluorescent cell-wall staining and section integrity; species example is Arabidopsis, NOT spinach.')]
    base.write_json(root/'research_sources.json',sources)
    mass=metrics['mass_fraction']
    (root/'REPORT_ZH.md').write_text(f'''# P12 第三版：不重叠的薄细胞切片

上版的问题是人为引入了细胞投影重叠，而且组织占据了20–80 µm共七层。这版按用户提供的真实样本约束重新生成。

- {len(items)} 个大小不一的圆润细胞，独立围出细胞腔。相邻细胞在接触处压平、缩进，允许小的细胞间隙。
- 每个几何子像素只属于一个细胞；在约 {STEP:.3f} µm 的XY几何网格上，细胞区域重叠计数为0，同一XY也不会放到另一深度再叠一颗细胞。
- 接触邻居共 {audit['touching_pairs']} 对；逐细胞检查：细胞区域连通、细胞腔连通、细胞腔密度为0。接触处保留互补边界，不把环形轮廓交叉后取最大值掩盖穿插。
- 切片局部几何支撑厚度14 µm，边缘使用平滑渐变，中心面只有轻微起伏。十层GT仅40、50、60 µm非零，三层能量分别 {100*mass[3]:.1f}%、{100*mass[4]:.1f}%、{100*mass[5]:.1f}%；其他七层严格为0。
- 原生GT为260×260×10；细网格Z采样1 µm、XY每像素3×3子采样积分。接触处一个原生像素可能包含两个邻居的边界，这是积分，不代表几何穿插。
- 所有预览从实际浮点真值绘制，共用整卷亮度上限；没有模拟光学模糊、散斑或噪声。

## 调研如何影响这版

菠菜专属研究的根横切面图注描述了中心木质部等组织分区，但仅凭用户这张图无法确定成像的是哪一段根、哪一组织区域及其物理尺寸，所以本轮不凭空添加完整的中心维管结构。[Wahua 与 Agogbua, 2023](https://www.ajol.info/index.php/sa/article/download/263438/248674)

植物细胞壁成像研究把细胞壁描述为包围细胞内容物的边界，展示了细胞腔、相邻边界和细胞间隙；它也说明染色及壁成分会影响荧光强度。这支持本版的接触边界与亮度差异。该研究的示例包括红树等植物，不是菠菜专属形态依据。[Kitin 等, 2020](https://www.fpl.fs.usda.gov/documnts/pdf2020/fpl_2020_kitin001.pdf)

根横切片制备资料强调避免压皱和损伤、清楚观察细胞壁。薄至最多三层的约束直接来自用户，14 µm局部厚度是本轮可审核的仿真设定，并非把文献中其他植物的厚度照搬为菠菜测量值。[Schiefelbein 实验室](https://sites.lsa.umich.edu/schiefelbein-lab/rapid-preparation-of-transverse-sections-of-plant-roots/)

## 预览与状态

[俯视图](P12/previews/P12_xy.png) · [原生十层](P12/previews/P12_native_layers.png) · [接触处与逐细胞标签](P12/previews/P12_contact_audit.png) · [新旧对比](P12/previews/P12_before_after.png)

当前仍待用户形态确认；原训练集及前两版预览保留。完整仿真只使用物理0号GPU，随后再发布经过验证的P12训练集扩展。
''')
    base.write_json(root/'preview_artifacts_sha256.json',{str(q.relative_to(root)):base.digest(q) for q in root.rglob('*') if q.is_file()})
    print(json.dumps(dict(output=str(root),metrics=metrics),ensure_ascii=False))


if __name__=='__main__':
    main()
