from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import tifffile

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
DATA = ROOT / 'outputs/v3_mean_anchor_e3_mean100_400_20260909_run01/dataset_v3_frozen_view'
REAL = ROOT / 'data/spinach_real_mixed_training_20260910_run01'
PSF = ROOT / 'data/.cache/psf/selected_H_d57d0714b548a9d8c888.npy'
SPARSE = ROOT / 'outputs/parallel_validation_scale_cov_mean_20260907/sparse_operator'
SPLITS = {'train': [f'P{i:02d}' for i in range(1,12)], 'validation': ['V01','V02','V03'], 'test':['T02','T03','T04']}
Z = list(range(10,101,10))


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


def keys(split):
    return [(obj, s) for obj in SPLITS[split] for s in range(1,11)]


def raw_mean(obj, subset):
    folder = DATA / obj
    with h5py.File(folder / 'subsets' / f'subset_{subset:02d}.mat') as f:
        ids = np.asarray(f['input_indices']).ravel().astype(int)
        rest = np.asarray(f['holdout_indices']).ravel().astype(int)
        z = np.asarray(f['z_um']).ravel()
        saved = np.asarray(f['input_physics_mean_float'], dtype=np.float32).T
    if len(set(ids)) != 10 or len(set(rest)) != 90 or set(ids) & set(rest) or set(ids)|set(rest) != set(range(1,101)):
        raise ValueError('Invalid 10/90 partition')
    if not np.allclose(z, Z, atol=1e-4, rtol=0):
        raise ValueError('Wrong depth grid')
    frames = []
    for i in ids:
        with h5py.File(folder / 'sensor_frames' / f'frame_{i:03d}.mat') as f:
            frames.append(np.asarray(f['sensor_pre_detector'], dtype=np.float32).T)
    mean = np.stack(frames).mean(0, dtype=np.float64).astype(np.float32)
    if not np.array_equal(mean, saved):
        raise ValueError(f'Saved mean differs: {obj}/{subset}')
    if mean.shape != (260,260) or not np.isfinite(mean).all() or np.any(mean < 0):
        raise ValueError('Invalid input image')
    return mean, ids.tolist()


def truth(obj):
    if obj not in sum(SPLITS.values(), []):
        raise ValueError('Unknown object')
    with h5py.File(DATA / obj / 'prepared.mat') as f:
        value = np.asarray(f['ground_truth'], dtype=np.float32).transpose(0,2,1).copy()
    if value.shape != (10,260,260) or not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError('Invalid GT')
    return value


def sample(obj, subset, *, supervised=False):
    path = BASE / 'cache/means' / f'{obj}_{subset:02d}.npy'
    if not path.is_file():
        raise FileNotFoundError('Run scripts/prepare_data.sh first')
    result = {'mean':np.load(path), 'object':obj, 'subset':subset}
    if supervised:
        result['truth'] = truth(obj)
    return result


def real_sample(field, subset):
    if field not in ('45','55') or subset not in range(1,11):
        raise ValueError('Real field must be 45 or 55; subset 1..10')
    manifest = json.loads((REAL / field / f'subset_{subset:02d}' / 'manifest.json').read_text())
    frames = np.stack([tifffile.imread(p).astype(np.float32) for p in manifest['selected_files']])
    if frames.shape != (10,1029,1421):
        raise ValueError(f'Unexpected real frames {frames.shape}')
    mean = frames.mean(0,dtype=np.float64).astype(np.float32)
    if not np.array_equal(mean, tifffile.imread(manifest['mean_tiff'])):
        raise ValueError('Real mean mismatch')
    return {'mean':mean, 'object':f'real_{field}', 'subset':subset}


def contract():
    path = BASE / 'manifests/data.json'
    if not path.is_file():
        raise FileNotFoundError('Run prepare_data.sh first')
    info = json.loads(path.read_text())
    if info['split_sha256'] != digest(DATA/'dataset_splits.json'):
        raise ValueError('Source split changed since preparation')
    return info


def prepare():
    declared = json.loads((DATA/'dataset_splits.json').read_text())
    actual = {s:[x['sample_id'] for x in declared['samples'] if x['split']==s] for s in SPLITS}
    if actual != SPLITS or not declared['dataset_complete']:
        raise ValueError('Frozen split mismatch')
    folder = BASE/'cache/means'; folder.mkdir(parents=True,exist_ok=True)
    records=[]; input_max=0.
    for split in SPLITS:
        for obj, subset in keys(split):
            mean, ids = raw_mean(obj,subset)
            p=folder/f'{obj}_{subset:02d}.npy'; np.save(p,mean)
            if split=='train': input_max=max(input_max,float(mean.max()))
            records.append({'object':obj,'subset':subset,'split':split,'input_indices':ids,'mean_sha256':digest(p)})
    # GT is inspected only for the separately declared VCD normalization, never by SeReNet training.
    target_scale=max(float(truth(obj).max()) for obj in SPLITS['train'])*1.05
    info={'version':1,'data_root':str(DATA),'split_sha256':digest(DATA/'dataset_splits.json'),
          'splits':SPLITS,'records':records,'input_scale':max(input_max,1e-12),
          'vcd_target_scale':max(target_scale,1e-12),
          'serenet_output_scale':max(input_max,1e-12)/float(np.load(SPARSE/'forward_data.npy',mmap_mode='r').sum(dtype=np.float64)/(260*260)),
          'z_um':Z,'pixel_pitch_um':220/49/4,
          'normalization':'fixed train maximum; VCD GT max * 1.05, separate from SeReNet',
          'psf_path':str(PSF),'psf_sha256':digest(PSF),'sparse_manifest':json.loads((SPARSE/'complete.json').read_text())}
    write_json(BASE/'manifests/data.json',info)
    print(json.dumps({'prepared':len(records),'input_scale':input_max,'vcd_target_scale':target_scale}),flush=True)
