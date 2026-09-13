"""Geometry-only P12 preview. This entry point cannot launch imaging or training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('MPLCONFIGDIR', '/tmp/root_cell_preview_mpl')
import numpy as np
from scipy.io import savemat, loadmat
from scipy.spatial import cKDTree
from scipy.ndimage import label
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils.experiment_paths import next_experiment_path

SOURCE = REPO / 'data/speckle_dataset_v3_full_20260907_run01'
REFERENCE = Path('/workspace/xyx/.codex/attachments/94dec1b2-d529-45ed-961f-3aae254e31ce/codex-clipboard-d0aa71aa-2585-491b-ae70-c215813299a3.png')
PITCH = 220 / 49 / 4


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def tissue(x, y):
    """Two asymmetric lobes, joined at a waist, with a peripheral scalloped edge."""
    yy = (y - 145) / 119
    width = 105 * np.sqrt(np.maximum(0, 1 - yy**2))
    width *= 1 - .20 * np.exp(-((y - 144) / 14)**2)
    width *= 1 + .045 * np.sin(y / 10) + .025 * np.cos(y / 5.7)
    mid = 145 + 7 * np.sin((y - 100) / 49)
    return (np.abs(x - mid) < width) & (np.abs(yy) < 1)


def seed_cells(seed):
    rng = np.random.default_rng(seed)
    sites, spacings = [], []
    # Variable-radius Poisson sampling: smaller cells around the perimeter.
    for _ in range(70000):
        point = rng.uniform([31, 27], [259, 263])
        if not tissue(*point):
            continue
        radius = np.sqrt(((point[0]-145)/105)**2 + ((point[1]-145)/119)**2)
        spacing = (9.0 + 4.7 * np.exp(-(radius/.65)**2)) * rng.uniform(.83, 1.20)
        if sites:
            distances = np.linalg.norm(np.asarray(sites) - point, axis=1)
            if np.any(distances < .5 * (np.asarray(spacings) + spacing)):
                continue
        sites.append(point)
        spacings.append(spacing)
        if len(sites) == 270:
            break
    return np.asarray(sites), rng.uniform(.48, 1, len(sites))


def geometry(seed):
    sites, amplitudes = seed_cells(seed)
    tree = cKDTree(sites)
    # Two lateral quadrature samples per native pixel; 1 um axial quadrature.
    axis = (np.arange(520)/2 - .25)*PITCH
    x, y = np.meshgrid(axis, axis)
    center_z = 53 + .055*(x-145) - .038*(y-145) + 6*np.sin((x-50)/47)*np.cos((y-100)/62)
    half_height = 6.5 + 1.8*np.sin(x/43)*np.cos(y/57)
    fine = np.zeros((260,260,100), np.float32)
    lumen_label = None
    for k, z in enumerate(np.arange(5.5,105,1)):
        dz = np.maximum(np.abs(z-center_z)-half_height, 0)
        gate = np.exp(-.5*(dz/1.2)**2)
        gate[dz > 3.6] = 0
        if not gate.any():
            continue
        # Smooth warp bends polygon walls in XY and Z without disconnected rings.
        u = x + 1.7*np.sin(y/12)*np.cos(x/38) + .040*(z-53)
        v = y + 1.5*np.cos(x/14)*np.sin(y/36) - .025*(z-53)
        distance, ids = tree.query(np.column_stack((u.ravel(),v.ravel())), k=4, workers=2)
        nearest = sites[ids[:,0]]
        separation = np.linalg.norm(sites[ids[:,1:]]-nearest[:,None,:],axis=2)
        # Distance to nearest Voronoi bisector, in physical micrometres.
        d = (distance[:,1:]**2-distance[:,0,None]**2)/(2*separation)
        choose = np.argmin(d,axis=1)
        wall_distance = np.take_along_axis(d,choose[:,None],axis=1)[:,0]
        other = ids[np.arange(len(ids)),choose+1]
        brightness = .5*(amplitudes[ids[:,0]]+amplitudes[other])
        wall = brightness*np.exp(-.5*(wall_distance/.8)**2)
        wall[wall_distance > 2.4] = 0
        wall = wall.reshape(520,520)*tissue(x,y)*gate
        # Prevent an artificial sealed fluorescent cap at top/bottom of the slice.
        fine[:,:,k] = wall.reshape(260,2,260,2).mean(axis=(1,3)).astype(np.float32)
        if k == 48:
            lumen_label = ids[:,0].reshape(520,520)[1::2,1::2]+1
    fine /= fine.max()
    coarse = fine.reshape(260,260,10,10).mean(axis=3,dtype=np.float32)
    return fine, coarse, sites, amplitudes, center_z, half_height, lumen_label


def validate(fine, coarse, inputs, holdout):
    assert fine.shape==(260,260,100) and coarse.shape==(260,260,10)
    assert fine.dtype==coarse.dtype==np.float32
    assert np.isfinite(fine).all() and np.isfinite(coarse).all() and fine.min()>=0
    assert fine.max()==1
    border = max(fine[[0,-1]].max(),fine[:,[0,-1]].max(),fine[:,:,[0,-1]].max())
    assert border==0
    expected = fine.reshape(260,260,10,10).mean(axis=3,dtype=np.float32)
    assert np.array_equal(expected,coarse)
    mass_error = abs(fine.sum(dtype=np.float64)-10*coarse.sum(dtype=np.float64))/fine.sum(dtype=np.float64)
    assert mass_error < 2e-6
    for a,b in zip(inputs,holdout):
        assert len(set(a)&set(b))==0 and len(set(a)|set(b))==100
    assert len(np.unique(inputs))==100
    masses = coarse.sum(axis=(0,1),dtype=np.float64)
    occupied = np.flatnonzero(masses)
    assert len(occupied)>=3 and np.all(np.diff(occupied)==1)
    assert not np.array_equal(coarse[:,:,occupied[0]],coarse[:,:,occupied[-1]])
    return dict(passed=True, array_axis_order='YXZ', fine_shape=list(fine.shape),
                coarse_shape=list(coarse.shape), border_max=float(border),
                fine_to_coarse_mass_relative_error=float(mass_error),
                occupied_z_um=((occupied+1)*10).tolist(),
                mass_fraction=(masses/masses.sum()).tolist(),
                fine_peak=float(fine.max()), coarse_peak=float(coarse.max()),
                fine_nonzero_z_range_um=[float(np.flatnonzero(fine.sum(axis=(0,1)))[0]+5.5),
                                        float(np.flatnonzero(fine.sum(axis=(0,1)))[-1]+5.5)],
                ten_ninety_indices_disjoint=True, forward_simulation_started=False)


def plots(folder, fine, coarse, sites):
    extent=[-PITCH/2,259.5*PITCH,259.5*PITCH,-PITCH/2]
    mip = coarse.max(axis=2)
    fig,ax=plt.subplots(figsize=(8,8),layout='constrained')
    ax.imshow(mip,cmap='inferno',vmin=0,vmax=coarse.max(),extent=extent,interpolation='nearest')
    ax.set(title='P12 | root-cell-wall-like ground truth | XY maximum',xlabel='X (um)',ylabel='Y (um)')
    fig.savefig(folder/'P12_xy.png',dpi=180); plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(14,9),layout='constrained')
    axs[0,0].imshow(mip,cmap='inferno',vmin=0,vmax=coarse.max(),extent=extent,interpolation='nearest')
    axs[0,0].set_title('XY max | native 10 um slabs')
    axs[0,1].imshow(coarse.max(axis=0).T,origin='lower',extent=[extent[0],extent[1],5,105],
                    cmap='inferno',vmin=0,vmax=coarse.max(),interpolation='nearest',aspect='equal')
    axs[0,1].set(xlabel='X (um)',ylabel='Z (um)',title='XZ max | native samples')
    axs[0,2].imshow(coarse.max(axis=1).T,origin='lower',extent=[extent[0],extent[1],5,105],
                    cmap='inferno',vmin=0,vmax=coarse.max(),interpolation='nearest',aspect='equal')
    axs[0,2].set(xlabel='Y (um)',ylabel='Z (um)',title='YZ max | native samples')
    mass=coarse.sum(axis=2)
    depth=(coarse*np.arange(10,101,10)[None,None,:]).sum(axis=2)/np.maximum(mass,1e-20)
    depth[mass<=.01*mass.max()]=np.nan
    im=axs[1,0].imshow(depth,extent=extent,cmap='viridis',vmin=10,vmax=100,interpolation='nearest')
    axs[1,0].set_title('Local mean depth (um)'); fig.colorbar(im,ax=axs[1,0],shrink=.7)
    masses=coarse.sum(axis=(0,1),dtype=np.float64)
    axs[1,1].bar(np.arange(10,101,10),100*masses/masses.sum(),width=8)
    axs[1,1].set(xlabel='Z (um)',ylabel='Mass (%)',title='Depth mass: no per-layer rescaling')
    axs[1,2].imshow(mip,cmap='gray',vmin=0,vmax=coarse.max(),extent=extent,interpolation='nearest')
    axs[1,2].set(xlim=(95,175),ylim=(135,65),title='Fixed 80 x 70 um wall detail')
    fig.suptitle('SYNTHETIC MORPHOLOGY ONLY | not sensor data | awaiting approval')
    fig.savefig(folder/'P12_overview.png',dpi=150); plt.close(fig)
    fig,axs=plt.subplots(2,5,figsize=(17,7),layout='constrained')
    for k,ax in enumerate(axs.flat):
        ax.imshow(coarse[:,:,k],cmap='inferno',vmin=0,vmax=coarse.max(),extent=extent,interpolation='nearest')
        ax.set_title(f'{10*(k+1)} um'); ax.axis('off')
    fig.suptitle('P12 | all 10 native GT slabs | ONE shared brightness scale')
    fig.savefig(folder/'P12_native_layers.png',dpi=160); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--seed',type=int,default=2026090901)
    args=parser.parse_args()
    manifests=[SOURCE/'dataset_splits.json',SOURCE/'FINAL_DATASET_MANIFEST.json']
    protected={str(p):digest(p) for p in manifests}
    split=json.loads(manifests[0].read_text())
    assert 'P12' not in [r['sample_id'] for r in split['samples']]
    root=args.output or next_experiment_path(REPO/'data','root_cell_P12_preview')
    root=root.resolve(); root.mkdir(parents=True,exist_ok=False)
    folder=root/'P12'; folder.mkdir(); preview=folder/'previews'; preview.mkdir()
    fine,coarse,sites,brightness,center,halfheight,labels=geometry(args.seed)
    rng=np.random.default_rng(2028090901)
    inputs=(rng.permutation(100)+1).reshape(10,10).astype(np.float64)
    holdout=np.array([np.setdiff1d(np.arange(1,101),a) for a in inputs],dtype=np.float64)
    metrics=validate(fine,coarse,inputs,holdout)
    cfg=dict(schema_version=4,geometry_version='root_cell_wall_slice_v1',
             dataset_id='P12_root_cell_preview',sample_id='P12',split='train',
             geometry_kind='root_like_cell_wall_slice',description='Synthetic root-cell-wall-like corrugated tissue slice',
             geometry_seed=args.seed,illumination_seed=2027090901,subset_seed=2028090901,
             image_size=np.array([260,260]),object_pixel_pitch_um=PITCH,
             x_um=np.arange(260)*PITCH,y_um=np.arange(260)*PITCH,
             z_um=np.arange(10,101,10),fine_dz_um=1,fine_z_um=np.arange(5.5,105,1),
             frame_count=100,input_frames=10,holdout_frames=90,subset_count=10,iterations=3,
             repo_root=str(REPO),output_root=str(root),sample_dir=str(folder),
             stage='truth_preview_only',morphology_approved=False,
             allowed_physical_gpu_indices=0,required_gpu_model='A40',preview_compute='CPU_ONLY',
             future_gpu_policy='physical GPU 0 only; no other GPU fallback',
             future_physics_policy='reuse source V3 acquisition PSF and illumination after morphology approval',
             lateral_quadrature_samples_per_pixel=2,wall_sigma_um=.8,
             wall_nominal_fwhm_um=2.35482*.8,wall_support_radius_um=2.4,
             axial_taper_sigma_um=1.2)
    meta=dict(array_axis_order='YXZ',geometry_axis_order='XYZ',geometry_units='um',
              normalization='one global fine-volume peak; no per-layer normalization',
              coarse_voxel_semantics='average density in centered 10 um slabs',
              fine_grid_semantics='1 um Z quadrature; native XY pixels integrated with 2x2 subpixels',
              morphology='smoothly warped Voronoi shared cell walls; open transverse tissue slice; dark lumens',
              biological_claim='stylized reference-inspired morphology; no calibrated anatomy or species-specific ground truth',
              sites_xy_um=sites,cell_amplitude=brightness,
              geometry=dict(kind='shared_wall_tessellation',site_count=len(sites),sites_xy_um=sites),
              reference_image_sha256=digest(REFERENCE) if REFERENCE.exists() else 'unavailable',
              source_dataset_manifest_sha256=protected[str(manifests[1])])
    source_sha=digest(__file__)
    savemat(folder/'truth.mat',dict(cfg=cfg,meta=meta,ground_truth=coarse,
            ground_truth_fine=fine,metrics=metrics,source_sha256=source_sha,
            approval_status='pending_user_morphology_review',input_indices=inputs,
            holdout_indices=holdout),do_compression=True,long_field_names=True)
    # Round-trip exactly the arrays which future MATLAB simulation will load.
    saved=loadmat(folder/'truth.mat')
    assert np.array_equal(saved['ground_truth'],coarse)
    assert np.array_equal(saved['ground_truth_fine'],fine)
    assert np.array_equal(saved['input_indices'],inputs)
    np.save(folder/'ground_truth_YXZ.npy',coarse)
    tifffile.imwrite(folder/'ground_truth_float.tif',coarse.transpose(2,0,1),metadata={'axes':'ZYX'},photometric='minisblack')
    tifffile.imwrite(folder/'ground_truth_fine_float.tif',fine.transpose(2,0,1),metadata={'axes':'ZYX'},photometric='minisblack')
    counts=np.bincount(labels[tissue(*np.meshgrid(np.arange(260)*PITCH,np.arange(260)*PITCH))].ravel())
    areas=counts[1:]*PITCH**2
    metrics['cell_site_count']=len(sites)
    metrics['equivalent_cell_diameter_um_percentiles_10_50_90']=np.percentile(2*np.sqrt(areas[areas>0]/np.pi),[10,50,90]).tolist()
    write_json(folder/'truth_metrics.json',metrics)
    write_json(folder/'geometry.json',dict(geometry_seed=args.seed,sites_xy_um=sites.tolist(),
               cell_amplitudes=brightness.tolist(),wall_sigma_um=.8,source_sha256=source_sha,
               reference_image_sha256=meta['reference_image_sha256'],
               fine_center_z_range_um=[float(center.min()),float(center.max())],
               open_cell_slice_without_fluorescent_caps=True))
    plots(preview,fine,coarse,sites)
    (root/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    assert protected=={str(p):digest(p) for p in manifests}
    contract=dict(stage='truth_preview_only',morphology_approved=False,simulation_started=False,
                  training_started=False,intended_split='train',intended_sample_id='P12',
                  source_dataset=str(SOURCE),source_dataset_manifests_unchanged=True,
                  protected_source_hashes=protected,allowed_physical_gpu_indices=[0],
                  no_gpu_fallback=True,preview_compute='CPU_ONLY',
                  future_plan='after explicit morphology approval simulate 100 frames, create ten 10/90 subsets and RL3; then publish validated training-set extension',
                  source_sha256=source_sha,truth_sha256=digest(folder/'truth.mat'),
                  matlab_axis_order='YXZ',tiff_axis_order='ZYX',metrics=metrics)
    write_json(root/'MORPHOLOGY_REVIEW_REQUIRED.json',contract)
    write_json(root/'dataset_extension_draft.json',dict(dataset_complete=False,
                  source_dataset=str(SOURCE),proposed_addition=dict(sample_id='P12',split='train',
                  sample_dir=str(folder),stage='truth_preview_only'),active_training_manifest_modified=False))
    report=f'''# P12 根细胞壁样本：真值待审核

这是一块参考图片形态构造的合成组织切片，细胞壁亮、细胞腔暗，细胞壁互相连接，外围细胞较小。它不是对参考图的三维反演，也不声称还原菠菜根的真实解剖结构。

- 拟加入训练集的新对象：P12；正式数据清单尚未更新。
- 细胞位置数：{len(sites)}；细胞等效直径的10/50/90百分位为 {metrics['equivalent_cell_diameter_um_percentiles_10_50_90']} µm。
- 壁厚设定：高斯截面 σ=0.8 µm，名义半高全宽约1.88 µm；这是几何尺寸，不是系统分辨率。
- 组织有起伏和厚度，细网格实际非零深度范围 {metrics['fine_nonzero_z_range_um']} µm；十层GT的非空层为 {metrics['occupied_z_um']} µm。
- 细网格沿Z为1 µm；正式GT为260×260×10。XY间距 {PITCH:.6f} µm；每个XY像素用2×2子采样积分。
- 只对整个细体归一化一次，再做十层slab平均；所有层图使用同一个亮度上限。
- 已通过非负、有限、边界不截断、slab质量守恒、MAT读回和10/90索引不重叠检查。
- 本次CPU预览没有使用GPU。完整仿真限定物理0号A40；只在你确认形态后启动。

请重点看：细胞是否过小或过密、壁是否过细、上下两片组织的轮廓是否合适，以及切片厚度和深度起伏是否符合你的需求。

[俯视图](P12/previews/P12_xy.png) · [深度与局部细节](P12/previews/P12_overview.png) · [原生十层](P12/previews/P12_native_layers.png)

后续完整仿真会沿用已有数据集的光学协议，保存100帧、十组10/90子集及RL3。原来的NA疑问不通过本轮几何预览擅自改动。P12是新几何类型，完整仿真时需新增对应验证与读取适配，不能直接套用旧入口的对象白名单。
'''
    (root/'REPORT_ZH.md').write_text(report)
    write_json(root/'preview_artifacts_sha256.json',{str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()})
    print(json.dumps(dict(output=str(root),metrics=metrics),ensure_ascii=False))


if __name__=='__main__':
    main()
