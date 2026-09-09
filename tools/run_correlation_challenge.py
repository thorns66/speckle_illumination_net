"""Additive frozen sensor-correlation benchmark, restricted to physical GPU 0.

Stages are checkpointed to disk. No training, original-data mutation or GPU reset.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import h5py
import numpy as np

from datasets.correlation_challenge import load_challenge_input, sha256
from datasets.matlab_multivolume_dataset import (
    DatasetItemKey, _read_targets, load_dataset_index, load_inference_input,
)
from tools.correlation_selection import correlation_matrix, score, select_extremes
from utils.experiment_paths import next_experiment_path

DATA = ROOT / 'data/speckle_dataset_v3_full_20260907_run01'
TRAIN = ROOT / 'outputs/v3_baseline_e3_mean005_400_20260908_run01'
PYTHON = Path(sys.executable)
MATLAB = Path('/workspace/xyx/MATLAB/R2023b/bin/matlab')
SAMPLES = ('T02', 'T03', 'T04')
ARMS = ('baseline', 'e3', 'e3_mean005')


def save(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    temporary.replace(path)


def log(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)


def gpu():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)
    fields = next(line.split(',') for line in text.splitlines() if line.split(',')[0].strip() == '0')
    return {'physical_index': 0, 'uuid': fields[1].strip(), 'free_mib': int(fields[2]),
            'utilization_percent': int(fields[3]), 'snapshot': text}


def wait_memory(output):
    while True:
        status = gpu()
        save(output / 'gpu_status.json', status)
        if status['free_mib'] >= 24000:
            return status
        log('Waiting for 24 GiB free on physical GPU 0; other GPUs excluded')
        time.sleep(30)


def prepare(output):
    path = output / 'run.json'
    if path.exists():
        return json.loads(path.read_text())
    _, fingerprint = load_dataset_index(DATA)
    status = gpu()
    challenge = next_experiment_path(DATA / 'test_challenges', 'correlation')
    challenge.mkdir(parents=True)
    manifest = {'version': 1, 'kind': 'sensor_correlation_test_challenge', 'complete': False,
                'source_dataset': str(DATA), 'source_fingerprint': fingerprint,
                'gpu_uuid': status['uuid'], 'physical_gpu': 0,
                'selection': {'seed': 20260908, 'random_candidates': 1000, 'starts_per_extreme': 100,
                              'objective': 'mean absolute spatial Pearson over 45 pairs',
                              'domain': 'sensor_pre_detector; full FOV; per-frame spatial centering',
                              'global_optimum_proven': False, 'allows_overlap': True},
                'samples': []}
    for name in SAMPLES:
        source = DATA / name
        directory = challenge / name
        (directory / 'subsets').mkdir(parents=True)
        (directory / 'sensor_frames').symlink_to(source / 'sensor_frames', target_is_directory=True)
        frames, hashes = [], {}
        for i in range(1, 101):
            frame = source / 'sensor_frames' / f'frame_{i:03d}.mat'
            hashes[str(frame)] = sha256(frame)
            with h5py.File(frame, 'r') as handle:
                frames.append(np.asarray(handle['sensor_pre_detector'], dtype=np.float32).T)
        for p in [source / 'prepared.mat'] + sorted((source / 'subsets').glob('*.mat')):
            hashes[str(p)] = sha256(p)
        frames = np.stack(frames)
        matrix = correlation_matrix(frames)
        selection = select_extremes(matrix)
        np.save(directory / 'correlation_signed.npy', matrix)
        np.save(directory / 'random_candidate_indices.npy', selection.pop('random_indices'))
        np.save(directory / 'random_candidate_scores.npy', selection.pop('random_scores'))
        random_scores = []
        for i in range(1, 11):
            with h5py.File(source / 'subsets' / f'subset_{i:02d}.mat') as h:
                indices = np.asarray(h['input_indices']).reshape(-1).astype(int) - 1
            random_scores.append({'subset_index': i, 'score': score(matrix, indices)})
        for group in ('low', 'high'):
            subset = frames[np.array(selection[group]['input_indices']) - 1].astype(np.float64)
            brightness = subset.mean(axis=(1, 2))
            selection[group]['brightness'] = {
                'mean_frame_intensity': float(brightness.mean()),
                'frame_intensity_cv': float(brightness.std() / brightness.mean()),
                'mean_spatial_contrast': float((subset.std(axis=(1, 2)) / brightness).mean()),
                'total_intensity': float(subset.sum()),
            }
            ids = np.array(selection[group]['input_indices']) - 1
            block = matrix[np.ix_(ids, ids)][np.triu_indices(10, 1)]
            selection[group]['signed_mean'] = float(block.mean())
        sample = {'sample_id': name, 'source_dir': str(source), 'challenge_dir': str(directory),
                  'source_hashes': hashes, 'original_random_subsets': random_scores,
                  'overlap_count': len(set(selection['low']['input_indices']) & set(selection['high']['input_indices'])),
                  **selection}
        manifest['samples'].append(sample)
        log(f"Selected {name}: low={selection['low']['score']:.6f}, high={selection['high']['score']:.6f}")
    manifest_path = challenge / 'manifest.json'
    save(manifest_path, manifest)
    run = {'output': str(output), 'manifest': str(manifest_path), 'source_experiment': str(TRAIN),
           'physical_gpu': 0, 'gpu_uuid': status['uuid'], 'initial_gpu_status': status,
           'dataset_fingerprint': fingerprint, 'checkpoints': {}, 'source_snapshot': {}}
    for arm in ARMS:
        for role, filename in [('best','checkpoint_best.pt'),('final','checkpoint_last.pt')]:
            p = TRAIN / arm / filename
            run['checkpoints'][f'{arm}_{role}'] = {'path': str(p), 'sha256': sha256(p)}
    snapshot = output / 'source_snapshot'
    snapshot.mkdir()
    # Freeze the entire Python implementation needed by the established adapters.
    paths = list(ROOT.glob('*.py'))
    for folder in ('tools','datasets','models','physics','losses','training','utils'):
        paths.extend((ROOT / folder).rglob('*.py'))
    paths.extend((ROOT / 'matlab_code').rglob('*.m'))
    for p in paths:
        dest = snapshot / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        run['source_snapshot'][str(p)] = sha256(p)
    save(path, run)
    return run


def verify_sources(run):
    for path, expected in run['source_snapshot'].items():
        if sha256(path) != expected:
            raise RuntimeError(f'Implementation changed during experiment: {path}')
    for checkpoint in run['checkpoints'].values():
        if sha256(checkpoint['path']) != checkpoint['sha256']:
            raise RuntimeError('Frozen checkpoint changed')
    manifest = json.loads(Path(run['manifest']).read_text())
    for sample in manifest['samples']:
        for path, expected in sample['source_hashes'].items():
            if sha256(path) != expected:
                raise RuntimeError(f'Original data changed: {path}')
    if load_dataset_index(DATA)[1] != run['dataset_fingerprint']:
        raise RuntimeError('Original dataset fingerprint changed')


def reconstruct(output, run):
    marker = output / 'rl_complete.json'
    if marker.exists():
        return
    wait_memory(output)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = run['gpu_uuid']
    env['MATLAB_PREFDIR'] = str(output / 'matlab_preferences')
    paths = [ROOT / 'matlab_code' / p for p in ('pilot_dataset','cell_dataset','Util','Solver')]
    quote = lambda p: "'" + str(p).replace("'", "''") + "'"
    expression = 'addpath(' + ','.join(map(quote, paths)) + ');pilot_build_correlation_challenge(' + quote(run['manifest']) + ');'
    with (output / 'rl.log').open('a') as stream:
        child = subprocess.Popen([str(MATLAB), '-singleCompThread', '-softwareopengl', '-batch', expression],
                                 cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        save(output / 'active_process.json', {'phase': 'RL3', 'pid': child.pid, 'physical_gpu': 0})
        log(f'MATLAB RL3 PID={child.pid}; see rl.log')
        if child.wait() != 0:
            raise RuntimeError('MATLAB RL3 failed; inspect rl.log')
    manifest = json.loads(Path(run['manifest']).read_text())
    regressions = []
    for sample in manifest['samples']:
        for index, group in enumerate(('low','high'), 1):
            subset = Path(sample['challenge_dir']) / 'subsets' / f'subset_{index:02d}.mat'
            sample[group]['subset_sha256'] = sha256(subset)
            raw = load_challenge_input(run['manifest'], sample['sample_id'], group)
            assert raw['f_var'].shape == (1, 10, 260, 260)
        with h5py.File(Path(sample['challenge_dir']) / 'regression.mat') as h:
            regressions.append({'sample': sample['sample_id'],
                                'mean_relative_l2': float(np.asarray(h['mean_relative_l2']).item()),
                                'taylor_relative_l2': float(np.asarray(h['taylor_relative_l2']).item())})
    manifest['complete'] = True
    save(run['manifest'], manifest)
    save(marker, {'complete': True, 'subsets': 6, 'physics_volumes': 12, 'regression': regressions})
    log('RL3 complete and original-subset regression passed')


def as_device(raw, device):
    import torch
    names = {'f_var','f_var_feature','g_mean','input_mean','residual_frames','z_values_um'}
    return {k: torch.from_numpy(v).float().unsqueeze(0).to(device) if k in names else v
            for k,v in raw.items()}


def infer(output, run):
    import torch
    from tools import v3_compare_experiment as exp
    from tools.v3_compare_evaluation import _common_scores
    from losses.self_supervised_losses import TaylorH2VarianceModel
    wait_memory(output)
    os.environ['CUDA_VISIBLE_DEVICES'] = run['gpu_uuid']
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    exp.configure_precision()
    manifest = json.loads(Path(run['manifest']).read_text())
    operator = None
    for arm in ARMS:
        for role in ('final','best'):
            checkpoint = run['checkpoints'][f'{arm}_{role}']
            payload = torch.load(checkpoint['path'], map_location='cpu', weights_only=False)
            config = payload['config']
            assert config['v3_compare']['kind'] == arm
            assert role != 'final' or payload['completed_steps'] == 400
            model = exp.build_model(config, initial=False).to(device)
            model.load_state_dict(payload['model_state'], strict=True)
            model.eval()
            if operator is None:
                operator = exp.load_operator(config, device)
            variance_model = TaylorH2VarianceModel(operator, **config['noise'])
            # Verify existing subset predictions before evaluating new inputs.
            regression = []
            for name in SAMPLES:
                item = as_device(load_inference_input(DATA/name,1),device)
                with torch.inference_mode():
                    result, _ = exp.forward(model,item,operator,config=config)
                pred = result.reconstruction[0,0].cpu().numpy()
                reference = np.load(TRAIN/'evaluation'/arm/role/f'{name}_subset_01'/'reconstruction.npy')
                error = float(np.linalg.norm(pred-reference)/max(np.linalg.norm(reference),1e-30))
                if error > 1e-5:
                    raise RuntimeError(f'Original inference mismatch: {arm}/{role}/{name}: {error}')
                regression.append({'sample': name, 'relative_l2': error})
            save(output / f'{arm}_{role}_regression.json', regression)
            for sample in manifest['samples']:
                for group in ('low','high'):
                    dest = output/'predictions'/arm/role/sample['sample_id']/group
                    dest.mkdir(parents=True, exist_ok=True)
                    if (dest/'complete.json').exists():
                        record = json.loads((dest/'complete.json').read_text())
                        assert record['checkpoint_sha256'] == checkpoint['sha256']
                        assert record['prediction_sha256'] == sha256(dest/'reconstruction.npy')
                        continue
                    raw = load_challenge_input(run['manifest'],sample['sample_id'],group)
                    item = as_device(raw,device)
                    with torch.inference_mode():
                        result, beta0 = exp.forward(model,item,operator,config=config)
                        prediction = result.reconstruction[0,0].cpu().numpy().copy()
                    # Ground truth/holdout never enter the model or frame selector.
                    key = DatasetItemKey(sample['sample_id'],1 if group=='low' else 2,'test_challenge',Path(sample['challenge_dir']))
                    target = _read_targets(key,include_ground_truth=False)
                    mean = torch.from_numpy(target['measured_mean']).unsqueeze(0).to(device)
                    variance = torch.from_numpy(target['measured_variance']).unsqueeze(0).to(device)
                    with torch.inference_mode():
                        common = _common_scores(result.reconstruction,mean,variance,operator,variance_model,config)
                    assert prediction.shape == (10,260,260) and np.isfinite(prediction).all() and prediction.min() >= 0
                    np.save(dest/'reconstruction.npy',prediction)
                    save(dest/'complete.json', {'complete': True, 'checkpoint_sha256': checkpoint['sha256'],
                         'weight_step': payload['completed_steps'], 'input_indices': raw['input_indices'].tolist(),
                         'prediction_sha256': sha256(dest/'reconstruction.npy'), 'common_scores': common})
                    log(f'Inference {arm}/{role}/{sample["sample_id"]}/{group}')
            del result, model, payload, item
            torch.cuda.empty_cache()
    # Same auxiliary physics metric definitions for both RL3 controls.
    controls = []
    for sample in manifest['samples']:
        for index, group in enumerate(('low','high'),1):
            raw = load_challenge_input(run['manifest'],sample['sample_id'],group)
            key = DatasetItemKey(sample['sample_id'],index,'test_challenge',Path(sample['challenge_dir']))
            target = _read_targets(key,include_ground_truth=False)
            mean = torch.from_numpy(target['measured_mean']).unsqueeze(0).to(device)
            variance = torch.from_numpy(target['measured_variance']).unsqueeze(0).to(device)
            for method,field in [('mean_rl3','g_mean'),('taylor_rl3','f_var')]:
                with torch.inference_mode():
                    metrics = _common_scores(torch.from_numpy(raw[field]).unsqueeze(0).to(device),mean,variance,
                                             operator,variance_model,config)
                controls.append({'sample_id':sample['sample_id'],'group':group,'method':method,**metrics})
    save(output/'control_physics.json',controls)
    save(output/'inference_complete.json',{'complete':True,'logical_predictions':36,
                                         'peak_memory_bytes':torch.cuda.max_memory_allocated()})
    log('GPU inference complete; exiting worker to release GPU before CPU reporting')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', choices=['infer','report'])
    args = parser.parse_args()
    output = args.output or next_experiment_path(ROOT/'outputs','v3_correlation_challenge')
    output.mkdir(parents=True,exist_ok=True)
    log(f'Output: {output}')
    try:
        if args.worker:
            run = json.loads((output/'run.json').read_text())
            verify_sources(run)
            if args.worker == 'infer':
                infer(output,run)
            else:
                from tools.report_correlation_challenge import report
                report(output,run)
            return
        run = prepare(output)
        verify_sources(run)
        reconstruct(output,run)
        for phase in ('infer','report'):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = run['gpu_uuid'] if phase=='infer' else ''
            env['MPLCONFIGDIR'] = str(output/'mpl_cache')
            with (output/f'{phase}.log').open('a') as stream:
                child = subprocess.Popen([str(PYTHON),str(Path(__file__).resolve()),'--output',str(output),'--worker',phase],
                                         cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                save(output/'active_process.json',{'phase':phase,'pid':child.pid,'physical_gpu':0 if phase=='infer' else None})
                if child.wait() != 0:
                    raise RuntimeError(f'{phase} failed; inspect {phase}.log')
        verify_sources(run)
        save(output/'complete.json',{'complete':True,'manifest':run['manifest'],'physical_gpu':0,
                                    'gpu_workers_exited':True,'final_gpu_status':gpu()})
        log('ALL COMPLETE')
    except Exception:
        save(output/'failure.json',{'traceback':traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
