"""Explicit additive test-only manifest reader; never changes training indexing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from datasets.matlab_multivolume_dataset import load_inference_input


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_challenge_input(manifest_path, sample_id, group):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('kind') != 'sensor_correlation_test_challenge' or group not in ('low', 'high'):
        raise ValueError('Invalid challenge manifest/group')
    sample = next(s for s in manifest['samples'] if s['sample_id'] == sample_id)
    expected = sample[group]
    for frame in expected['input_indices']:
        path = Path(sample['source_dir']) / 'sensor_frames' / f'frame_{frame:03d}.mat'
        if sha256(path) != sample['source_hashes'][str(path)]:
            raise ValueError(f'Changed source frame: {path}')
    path = Path(sample['challenge_dir']) / 'subsets' / f"subset_{1 if group == 'low' else 2:02d}.mat"
    if expected.get('subset_sha256') and sha256(path) != expected['subset_sha256']:
        raise ValueError('Changed challenge subset')
    result = load_inference_input(sample['challenge_dir'], 1 if group == 'low' else 2)
    if not np.array_equal(result['input_indices'], expected['input_indices']):
        raise ValueError('Challenge frame selection changed')
    result['challenge_group'] = group
    result['challenge_manifest'] = str(Path(manifest_path).resolve())
    return result
