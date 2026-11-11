"""Launch independent plan entries with GNU Parallel; run large jobs separately."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid


def cleanup_containers(batch: str) -> None:
    from predictive_fsc.baselines._docker import BATCH_LABEL, remove

    command = ['docker', 'ps', '--all', '--quiet', '--filter', f'label={BATCH_LABEL}={batch}']
    # The daemon may still be completing a create request after its client exits.
    for _ in range(3):
        names = subprocess.check_output(command, text=True, timeout=15).split()
        for name in names:
            remove(name)
        time.sleep(.2)
    if subprocess.check_output(command, text=True, timeout=15).strip():
        raise OSError(f'containers remain; inspect with: {shlex.join(command)}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--jobs', type=int, required=True,
                        help='maximum concurrent entries; choose after measuring RAM use')
    parser.add_argument('--output', type=Path, help='batch directory (default: outputs/<plan>-parallel)')
    parser.add_argument('--force', action='store_true', help='rerun completed results')
    parser.add_argument('--dry-run', action='store_true', help='validate and print commands without writing or running jobs')
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    plan = args.plan.resolve()
    output = (args.output or Path('outputs') / f'{plan.stem}-parallel').resolve()
    cli = [sys.executable, '-m', 'predictive_fsc', 'reproduce', str(plan)]
    try:
        # Validate all entries before any experiment starts. No Docker is used.
        resolved = subprocess.run([*cli, '--dry-run'], capture_output=True, text=True, check=True)
        rows = [json.loads(line) for line in resolved.stdout.splitlines()]
        commands = [shlex.join([*cli, '--run-index', str(index), '--output', str(output / f'job-{index}'),
                                *(['--force'] if args.force else [])])
                    for index in range(1, len(rows) + 1)]
        if args.dry_run:
            print('\n'.join(commands))
            return 0

        parallel = shutil.which('parallel')
        if parallel is None:
            parser.error('GNU Parallel is required (on Debian/Ubuntu: apt install parallel)')
        version = subprocess.check_output([parallel, '--version'], text=True)
        if not version.startswith('GNU parallel '):
            parser.error('parallel must be GNU Parallel')
        baselines = {row['solver'] for row in rows} - {'mapo'}
        for solver in sorted(baselines):
            subprocess.run(['docker', 'image', 'inspect', f'predictive-fsc-{solver}'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=15, check=True)

        # Job numbers must keep referring to the same unchanged plan on reruns.
        identity = dict(plan=str(plan), sha256=hashlib.sha256(plan.read_bytes()).hexdigest())
        marker = output / 'plan.json'
        if marker.exists() and json.loads(marker.read_text()) != identity:
            parser.error('this output directory belongs to a different plan; choose another --output')
        output.mkdir(parents=True, exist_ok=True)
        temporary = marker.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(identity, indent=2) + '\n')
        temporary.replace(marker)

        batch = uuid.uuid4().hex
        joblog = ('+' if (output / 'joblog.tsv').exists() else '') + 'joblog.tsv'
        command = [parallel, '--plain', '--jobs', str(args.jobs), '--null', '--no-run-if-empty',
                   '--joblog', joblog, '--results', 'job-{#}/',
                   '--term-seq', 'TERM,20000,KILL,1000']
        environment = dict(os.environ, PREDICTIVE_FSC_BATCH=batch, PARALLEL_SHELL='/bin/sh')
        process = None
        previous_int = signal.getsignal(signal.SIGINT)
        previous = signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
        print(f'Running {len(commands)} entries with at most {args.jobs} jobs; logs: {output}', file=sys.stderr)
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, text=True,
                                       env=environment, cwd=output, start_new_session=True)
            process.communicate('\0'.join(commands) + ('\0' if commands else ''))
            return process.wait()
        finally:
            # Let GNU Parallel terminate its children and allow adapter cleanup.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            try:
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                if baselines:
                    try:
                        cleanup_containers(batch)
                    except (OSError, subprocess.SubprocessError) as error:
                        raise OSError(f'batch cleanup failed; inspect docker ps -a --filter '
                                      f'label=predictive-fsc.batch={batch}: {error}') from error
            finally:
                signal.signal(signal.SIGINT, previous_int)
                signal.signal(signal.SIGTERM, previous)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(2, f'error: {getattr(error, "stderr", None) or error}\n')


if __name__ == '__main__':
    sys.exit(main())
