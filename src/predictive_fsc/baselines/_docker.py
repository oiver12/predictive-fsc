"""Scoped container ownership for the adapters and the external batch launcher."""

from collections.abc import Iterator
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
from threading import current_thread, main_thread


BATCH_ENV = 'PREDICTIVE_FSC_BATCH'
BATCH_LABEL = 'predictive-fsc.batch'


def labels() -> list[str]:
    batch = os.environ.get(BATCH_ENV)
    return ['--label', f'{BATCH_LABEL}={batch}'] if batch else []


def image_id(tag: str) -> str:
    """Identify a locally built solver image by its content ID."""
    return subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', tag],
                                   text=True, stderr=subprocess.PIPE).strip()


def run_command(name: str, tmpfs: str) -> list[str]:
    """Offline, read-only container running as the calling user."""
    return ['docker', 'run', '--rm', *labels(), '--name', name, '--network', 'none',
            '--read-only', '--tmpfs', tmpfs, '--user', f'{os.getuid()}:{os.getgid()}']


def write_result(path: Path, result: object) -> None:
    """Write result.json atomically, so its presence marks a completed job."""
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def remove(name: str) -> None:
    result = subprocess.run(['docker', 'rm', '--force', name], stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True, timeout=15, check=False)
    if result.returncode and 'No such container' not in result.stderr:
        raise OSError(f'could not remove container {name}: {result.stderr.strip()}')


@contextmanager
def cleanup(name: str) -> Iterator[None]:
    try:
        yield
    finally:
        # A second launcher signal must not interrupt the bounded removal call.
        previous = {}
        if current_thread() is main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, signal.SIG_IGN)
        try:
            remove(name)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
