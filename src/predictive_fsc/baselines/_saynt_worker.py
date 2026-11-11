"""Run SAYNT, collect native progress, and stop at the container CPU budget.

The host mounts this file into the existing engine image. The supervisor uses
only the standard library; its child loads PAYNT and its dependencies.
"""

import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def container_cpu_seconds():
    """CPU consumed by this container, including all solver descendants."""
    text = Path('/sys/fs/cgroup/cpu.stat').read_text()
    counters = dict(line.split() for line in text.splitlines())
    return int(counters['usage_usec']) / 1_000_000


def run_solver(directory):
    """Child entry point: seed Python/NumPy and run the installed PAYNT CLI."""
    import random
    import runpy
    import numpy as np

    settings = json.loads((directory / 'settings.json').read_text())
    seed = settings['seed'] % 2**32
    random.seed(seed)
    np.random.seed(seed)

    native = '/opt/saynt/source/paynt/paynt.py'
    sys.path.insert(0, str(Path(native).parent))
    wall_limit = math.ceil(10 * settings['budget_seconds'] + 60)
    sys.argv = [
        native,
        '--project', str(directory),
        '--sketch', 'sketch.drn',
        '--filetype', 'drn',
        '--fsc-synthesis',
        '--storm-pomdp',
        '--iterative-storm', str(wall_limit),
        str(settings['paynt_seconds']), str(settings['storm_seconds']),
        '--export-fsc-paynt', str(directory / 'latest'),
        '--export-fsc-storm', str(directory / 'latest'),
    ]
    runpy.run_path(native, run_name='__main__')


def read_progress(reader, reward_correction):
    """Read complete two-line PAYNT/Storm reports; wait for unfinished lines.

    Example native report:
        -----------PAYNT-----------
        Value = 812.76 | Time elapsed = 3.1s | FSC size = 89
    """
    records = []
    while True:
        start = reader.tell()
        header = reader.readline()
        controller = header.strip().strip('-').lower()
        if controller not in ('paynt', 'storm'):
            if not header.endswith('\n'):
                reader.seek(start)
                break
            continue

        line = reader.readline()
        if not line.endswith('\n'):
            reader.seek(start)
            break
        if not line.startswith('Value = '):
            continue

        value, elapsed, size = line.split('|')
        records.append({
            'controller': controller,
            'value': float(value.split('=')[1]) - reward_correction,
            'solver_seconds': float(elapsed.split('=')[1].strip().removesuffix('s')),
            'size': int(size.split('=')[1]),
        })
    return records


def main(directory):
    initial_cpu = container_cpu_seconds()
    started_at = time.monotonic()
    directory = Path(directory)
    settings = json.loads((directory / 'settings.json').read_text())
    metadata = json.loads((directory / 'conversion.json').read_text())
    budget = settings['budget_seconds']
    reward_correction = metadata['horizon'] * metadata['reward_shift']

    def charged_cpu():
        return settings['setup_cpu_seconds'] + container_cpu_seconds() - initial_cpu

    # Forced reruns must not reuse the previous solver's exports.
    for path in (directory / 'latest').glob('*.fsc'):
        path.unlink()

    command = [sys.executable, '-u', __file__, str(directory), '--solver']
    trace = []
    best = None
    process = None
    status = 'budget'
    log_path = directory / 'solver.log'
    with log_path.open('w') as log, log_path.open() as reader:
        if charged_cpu() < budget:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        try:
            while process is not None and charged_cpu() < budget:
                records = read_progress(reader, reward_correction)
                captured_cpu = charged_cpu()
                for record in records:
                    if captured_cpu > budget:
                        continue
                    record['charged_cpu'] = captured_cpu
                    trace.append(record)
                    if best is not None and record['value'] <= best['value']:
                        continue

                    best = record.copy()
                    export = directory / 'latest' / (record['controller'] + '.fsc')
                    best['policy'] = None
                    if export.exists():
                        shutil.copyfile(export, directory / 'best.fsc')
                        best['policy'] = 'best.fsc'
                        best['policy_cpu'] = charged_cpu()

                if process.poll() is not None:
                    status = 'completed' if process.returncode == 0 else 'error'
                    break
                if time.monotonic() - started_at > 10 * budget + 60:
                    status = 'wall_guard'
                    break
                time.sleep(.02)
        finally:
            if process is not None:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    if 'Traceback (most recent call last)' in log_path.read_text():
        status = 'error'
    result = {
        'status': status,
        'returncode': process.returncode if process is not None else None,
        'value': None,
        'size': None,
        'policy': None,
        'policy_cpu': None,
        'trace': trace,
        'charged_cpu': charged_cpu(),
        'charged_wall': time.monotonic() - started_at,
        'value_source': 'solver_reported',
    }
    if best is not None:
        for field in ('value', 'size', 'policy', 'policy_cpu'):
            result[field] = best.get(field)
    (directory / 'worker-result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    directory = Path(sys.argv[1])
    if '--solver' in sys.argv[2:]:
        run_solver(directory)
    else:
        main(directory)
