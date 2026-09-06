#!/usr/bin/env python3
"""Launch one frozen MATLAB comparison per idle GPU, without dataset writes."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scope', choices=['full100', 'subsets10'], default='full100')
    parser.add_argument('--gpus', default='0,1,2,3,4,5')
    parser.add_argument('--allow-busy-gpus', action='store_true',
                        help='Explicit opt-in to shared GPUs; requires sufficient free memory')
    parser.add_argument('--min-free-gib', type=float, default=24.0)
    parser.add_argument('--max-workers', type=int, default=6)
    parser.add_argument('--subset-chunk-size', type=int, default=10,
                        help='Recheck idle GPU preference between chunks; 1 checks every subset')
    parser.add_argument('--matlab', default='/workspace/xyx/MATLAB/R2023b/bin/matlab')
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    script = Path(__file__).resolve().with_suffix('.m')
    assert script.is_file(), script
    gpu_ids = [int(item) for item in args.gpus.split(',')]
    assert len(gpu_ids) == len(set(gpu_ids)) and len(gpu_ids) > 0
    inventory = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu,memory.total',
        '--format=csv,noheader,nounits'], text=True)
    gpu_rows = {}
    for line in inventory.strip().splitlines():
        index, uuid, memory, utilization, total = [part.strip() for part in line.split(',')]
        gpu_rows[int(index)] = (uuid, int(memory), int(utilization), int(total))
    for gpu in gpu_ids:
        assert gpu in gpu_rows, f'GPU {gpu} is unavailable'
        assert (gpu_rows[gpu][3] - gpu_rows[gpu][1]) >= args.min_free_gib * 1024, (
            f'GPU {gpu} has less than {args.min_free_gib} GiB free memory')
        if not args.allow_busy_gpus:
            assert gpu_rows[gpu][1] < 1024 and gpu_rows[gpu][2] <= 10, (
                f'GPU {gpu} is busy; refusing to launch without explicit opt-in')
    data_root = repo / 'data/matlab_cells_pilot_v2_r04'
    assert output != data_root and data_root not in output.parents
    manifest = data_root / 'dataset_splits.json'
    splits = json.loads(manifest.read_text())
    objects = [row['sample_id'] for row in splits['samples'] if row['split'] == 'test']
    assert objects == ['P07', 'T01', 'T02']
    tasks = [(obj, mode) for obj in objects for mode in ['mean', 'taylor']]
    expected_tags = ['full100'] if args.scope == 'full100' else [f'subset_{i:02d}' for i in range(1, 11)]
    for obj, mode in tasks:
        for tag in expected_tags:
            folder = output / obj / tag / mode
            assert not folder.exists(), f'Refusing to overwrite {folder}'
    source_paths = [manifest, data_root / 'FINAL_DATASET_MANIFEST.json', script,
                    Path(__file__).resolve(), repo / 'matlab_code/Solver/deconvRL.m',
                    repo / 'matlab_code/pilot_dataset/pilot_reconstruct_volume.m',
                    repo / 'matlab_code/pilot_dataset/pilot_deconv_rl_nonnegative.m']
    for obj in objects:
        source_paths += [data_root / obj / 'validation_manifest.json']
        source_paths += sorted((data_root / obj / 'sensor_frames').glob('frame_*.mat'))
        source_paths += sorted((data_root / obj / 'subsets').glob('subset_*.mat'))
    hashes = {str(path): sha256(path) for path in source_paths}
    output.mkdir(parents=True, exist_ok=True)
    logs = output / 'logs'; logs.mkdir(exist_ok=True)
    assert 1 <= args.subset_chunk_size <= 10 and args.max_workers > 0
    jobs = [(obj, mode, list(range(first, min(first + args.subset_chunk_size, 11))))
            for obj, mode in tasks for first in range(1, 11, args.subset_chunk_size)] if args.scope == 'subsets10' else [(obj, mode, [0]) for obj, mode in tasks]
    contract = {'scope': args.scope, 'objects': objects, 'methods': ['mean', 'taylor'],
                'iterations': 5, 'gpu_inventory': inventory, 'source_sha256': hashes,
                'network_reference': 'multivolume_n10_no_mean_run01/checkpoint_best.pt step 160',
                'expected_reconstructions': len(tasks) * len(expected_tags),
                'status': 'running', 'started_unix': time.time(),
                'shared_gpus_explicitly_allowed': args.allow_busy_gpus,
                'minimum_free_memory_gib': args.min_free_gib,
                'gpu_policy': 'prefer idle cards at every chunk boundary; sharing only with explicit opt-in',
                'subset_chunk_size': args.subset_chunk_size, 'max_workers': args.max_workers}
    contract_path = output / f'run_{args.scope}.json'
    assert not contract_path.exists(), contract_path
    contract_path.write_text(json.dumps(contract, indent=2) + '\n')

    def matlab_string(value):
        return "'" + str(value).replace("'", "''") + "'"

    def worker(gpu, assigned):
        results = []
        for obj, mode, subsets in assigned:
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu_rows[gpu][0]
            suffix = f'_{subsets[0]:02d}_{subsets[-1]:02d}' if args.scope == 'subsets10' else ''
            pref = output / 'matlab_prefs' / f'{args.scope}_{obj}_{mode}{suffix}'
            pref.mkdir(parents=True, exist_ok=True)
            env['MATLAB_PREFDIR'] = str(pref)
            log = logs / f'{args.scope}_{obj}_{mode}{suffix}.log'
            expr = (f'addpath({matlab_string(script.parent)}); '
                    f'issues=checkcode({matlab_string(script)},"-id"); '
                    f'assert(~any(strcmp({{issues.id}},"PARSE")),"MATLAB parse error"); '
                    f'run_test_rl5_baselines({matlab_string(repo)},{matlab_string(output)},'
                    f'{matlab_string(obj)},{matlab_string(mode)},{matlab_string(args.scope)},'
                    f'[{" ".join(str(i) for i in subsets)}]);')
            print(f'LAUNCH {obj} {mode} {args.scope} subsets {subsets} physical GPU {gpu}', flush=True)
            started = time.monotonic()
            with log.open('x') as stream:
                run = subprocess.run([args.matlab, '-batch', expr], cwd=repo, env=env,
                                     stdout=stream, stderr=subprocess.STDOUT)
            result = {'sample_id': obj, 'method': mode, 'gpu': gpu,
                      'subsets': subsets,
                      'returncode': run.returncode, 'seconds': time.monotonic() - started,
                      'log': str(log)}
            print(json.dumps(result), flush=True)
            results.append(result)
        return results

    workers = min(len(gpu_ids), len(jobs), args.max_workers)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        pending, active = list(jobs), {}
        while pending or active:
            reserved = set(active.values())
            snapshot = subprocess.check_output([
                'nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu,memory.total',
                '--format=csv,noheader,nounits'], text=True)
            candidates = []
            for line in snapshot.strip().splitlines():
                gpu, used, utilization, total = map(int, line.split(','))
                idle = used < 1024 and utilization <= 10
                if gpu in gpu_ids and gpu not in reserved and total - used >= args.min_free_gib * 1024 and (idle or args.allow_busy_gpus):
                    candidates.append((not idle, used, utilization, gpu))
            for busy, used, utilization, gpu in sorted(candidates):
                if not pending or len(active) >= workers:
                    break
                print(f'SCHEDULE GPU {gpu}: {"shared" if busy else "idle"}, used={used} MiB, utilization={utilization}%', flush=True)
                active[executor.submit(worker, gpu, [pending.pop(0)])] = gpu
            if active:
                done, _ = concurrent.futures.wait(active, timeout=10, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    results.extend(future.result())
                    del active[future]
            elif pending:
                time.sleep(10)
    changed = [path for path, digest in hashes.items() if sha256(path) != digest]
    complete_files = [output / obj / tag / mode / 'complete.json'
                      for obj, mode in tasks for tag in expected_tags]
    success = (all(result['returncode'] == 0 for result in results) and not changed
               and all(path.is_file() for path in complete_files))
    contract.update(status='complete' if success else 'failed', results=results,
                    source_files_unchanged=not changed, changed_source_files=changed,
                    finished_unix=time.time())
    contract_path.write_text(json.dumps(contract, indent=2) + '\n')
    if not success:
        raise SystemExit('Comparison failed; inspect run manifest and individual logs')
    print(f'COMPLETE {len(complete_files)} reconstructions, source hashes unchanged', flush=True)


if __name__ == '__main__':
    main()
