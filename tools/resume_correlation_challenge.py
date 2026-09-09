"""Resume frozen RL3 challenge with transitive dependency verification."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from tools import run_correlation_challenge as original
from tools.correlation_dependency_guard import scoped_run
from datasets.correlation_challenge import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['infer', 'report'])
    args = parser.parse_args()
    out = args.output.resolve()
    run = json.loads((out / 'run.json').read_text())
    assert run['physical_gpu'] == 0
    entries = [ROOT / 'tools' / n for n in ('run_correlation_challenge.py',
                'report_correlation_challenge.py', 'v3_compare_experiment.py')]
    checked = scoped_run(run, ROOT, entries)
    original.verify_sources(checked)
    original.save(out / 'dependency_verification.json', {
        'complete': True, 'policy': 'transitive static local imports including lazy imports and package initializers',
        'checked': checked['source_snapshot'],
        'excluded_unrelated': sorted(set(run['source_snapshot']) - set(checked['source_snapshot'])),
        'repair_sources': {str(p):sha256(p) for p in [Path(__file__).resolve(),ROOT/'tools/correlation_dependency_guard.py']},
        'data_and_checkpoint_guards_unchanged': True,
    })
    if args.phase:
        if args.phase == 'infer':
            original.infer(out, run)
        else:
            from tools.report_correlation_challenge import report
            report(out, run)
        original.verify_sources(checked)
        return
    assert json.loads((out / 'rl_complete.json').read_text())['complete']
    # Preserve the old failure; it remains an audit record, not current status.
    original.save(out / 'resume_status.json', {'state': 'running', 'pid': os.getpid(), 'physical_gpu': 0})
    try:
        for phase in ('infer', 'report'):
            marker = out / ('inference_complete.json' if phase == 'infer' else 'report_complete.json')
            if marker.exists() and json.loads(marker.read_text()).get('complete'):
                continue
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = run['gpu_uuid'] if phase == 'infer' else ''
            env['MPLCONFIGDIR'] = str(out / 'mpl_cache')
            with (out / f'{phase}_resume.log').open('a') as stream:
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                    '--output', str(out), '--phase', phase], cwd=ROOT, env=env,
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                original.save(out / 'active_process.json', {'phase':phase,'pid':process.pid,
                    'physical_gpu':0 if phase=='infer' else None})
                if process.wait() != 0:
                    raise RuntimeError(f'{phase} failed; see {phase}_resume.log')
        original.verify_sources(checked)
        original.save(out / 'complete.json', {'complete':True,'gpu_workers_exited':True,
            'physical_gpu':0,'manifest':run['manifest'],'final_gpu_status':original.gpu(),
            'historical_failure_resolved':True})
        original.save(out / 'resume_status.json', {'state':'complete','gpu_workers_exited':True})
        original.log('ALL COMPLETE')
    except Exception:
        original.save(out / 'resume_status.json', {'state':'failed','traceback':traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
