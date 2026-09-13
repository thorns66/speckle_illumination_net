"""Wait for the existing Taylor workers, then run the authorized Mean-800 arm.

The user service manager owns this supervisor. It never launches, signals or
resumes Taylor, and leaves the frozen training/queue source files unchanged.
"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from tools import queue_v5_taylor_then_mean_800 as queue


def taylor_workers() -> list[int]:
    result = []
    worker = str(queue.TAYLOR_WORKER).encode()
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if worker in args and b'--worker' in args:
            result.append(int(entry.name))
    return sorted(result)


def main() -> None:
    with (queue.MEAN_OUTPUT / 'handoff_supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        preflight = json.loads((queue.MEAN_OUTPUT / 'preflight.json').read_text())
        queue.verify_frozen_sources(preflight)
        for path, expected in preflight['config_hashes'].items():
            if queue.old.sha256(path) != expected:
                raise ValueError(f'Frozen Mean config changed: {path}')
        queue.update_status(queue.QUEUE_STATUS, 'waiting_for_existing_taylor_800',
                            supervisor_pid=os.getpid(),
                            supervisor='systemd user service',
                            service='speckle-v5-mean800-handoff-20260911.service')
        print('Waiting for existing Taylor workers; no Taylor process will be changed.', flush=True)
        while True:
            workers = taylor_workers()
            complete = queue.completed_exactly(queue.TAYLOR_COMPLETE, queue.TARGET_STEPS)
            if complete and not workers:
                break
            if not workers and not complete:
                raise RuntimeError('Taylor workers exited without a step-800 completion marker')
            queue.update_status(queue.QUEUE_STATUS, 'waiting_for_existing_taylor_800',
                                taylor_worker_pids=workers,
                                taylor_training_complete=complete)
            time.sleep(20)
        if queue.checkpoint_steps(queue.TAYLOR_CHECKPOINT) != queue.TARGET_STEPS:
            raise ValueError('Taylor checkpoint does not contain exactly 800 updates')
        if not (queue.TAYLOR_DIR / 'checkpoint_step_000800.pt').is_file():
            raise FileNotFoundError('Taylor step-800 archival checkpoint is missing')
        queue.update_status(queue.TAYLOR_STATUS, 'complete', completed_steps=800,
                            verified_by='systemd handoff', finished_unix=time.time())
        print('Taylor 800 completed and workers released; starting Mean 800.', flush=True)
        queue.train_mean()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        queue.update_status(queue.QUEUE_STATUS, 'handoff_failed', reason=str(error))
        raise
