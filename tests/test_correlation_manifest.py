import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from datasets.correlation_challenge import load_challenge_input, sha256


class CorrelationManifestTests(unittest.TestCase):
    def test_manifest_reader_rejects_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'T02'
            (source / 'sensor_frames').mkdir(parents=True)
            hashes = {}
            for i in range(1, 11):
                p = source / 'sensor_frames' / f'frame_{i:03d}.mat'
                p.write_bytes(b'fixture')
                hashes[str(p)] = sha256(p)
            manifest = Path(temporary) / 'manifest.json'
            manifest.write_text(json.dumps({'kind':'sensor_correlation_test_challenge','samples':[
                {'sample_id':'T02','source_dir':str(source),'challenge_dir':str(source),
                 'source_hashes':hashes,'low':{'input_indices':list(range(1,11))}}]}))
            with patch('datasets.correlation_challenge.load_inference_input') as reader:
                reader.return_value = {'input_indices':np.arange(1,11)}
                self.assertEqual(load_challenge_input(manifest,'T02','low')['challenge_group'],'low')
                reader.return_value = {'input_indices':np.arange(2,12)}
                with self.assertRaises(ValueError):
                    load_challenge_input(manifest,'T02','low')
                reader.return_value = {'input_indices':np.arange(1,11)}
                p.write_bytes(b'changed')
                with self.assertRaises(ValueError):
                    load_challenge_input(manifest,'T02','low')
