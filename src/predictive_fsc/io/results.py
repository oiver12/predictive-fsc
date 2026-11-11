"""Portable numerical run identity and one-file results, without pickle."""

from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess

import numpy as np

from .. import _em
from ..dynamics import canonical
from ..types import (
    Controller, EMConfig, Problem, RunProvenance, SearchConfig, SearchFailure,
    SearchProgress, SearchResult,
)


def provenance(problem: Problem, seed: int) -> RunProvenance:
    """Hash the numerical problem, including horizon and explicit terminal rules."""
    digest = hashlib.sha256()

    def add(array, dtype):
        array = np.ascontiguousarray(array, dtype=dtype)
        digest.update(repr(array.shape).encode())
        digest.update(array.tobytes())

    model = problem.model
    add([problem.horizon, model.num_states, model.num_actions,
         model.num_observations], '<i8')
    for source in model.transitions:
        matrix = canonical(source)
        add(matrix.indptr, '<i8')
        add(matrix.indices, '<i8')
        add(matrix.data, '<f8')
    for array in (model.observation_probabilities, model.rewards, model.initial_belief):
        add(array, '<f8')
    add(sorted(model.terminal_actions), '<i8')
    add(sorted(model.terminal_states), '<i8')
    package = Path(__file__).parent.parent
    revision = 'unknown'
    repository = package.parent.parent
    if (repository / '.git').exists() and (repository / 'src' / 'predictive_fsc') == package:
        try:
            revision = subprocess.check_output(
                ['git', '-C', str(repository), 'rev-parse', 'HEAD'],
                text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    sources = hashlib.sha256()
    for path in sorted(package.rglob('*.py')):
        sources.update(path.relative_to(package).as_posix().encode())
        sources.update(path.read_bytes())
    build = {name: version(name) for name in ('predictive-fsc', 'numpy', 'scipy', 'numba', 'matplotlib')}
    build.update(python=platform.python_version(), platform=platform.platform(), compiler=_em.compiler,
                 native_sha256=hashlib.sha256(Path(_em.__file__).read_bytes()).hexdigest(),
                 python_source_sha256=sources.hexdigest(), problem_name=problem.name)
    # Container images have no Git checkout; the Dockerfile passes the revision in.
    if revision == 'unknown':
        revision = os.environ.get('PREDICTIVE_FSC_REVISION', revision)
    return RunProvenance(digest.hexdigest(), revision, seed, build)


def save_result(path: str | Path, result: SearchResult) -> None:
    """Save controller arrays and JSON metadata in one compressed NPZ archive.

    Metadata includes settings, independent seeds, all candidate/EM traces,
    failures, actual CPU use and numerical/source/build identity. No Python
    classes are pickled. Write results after solving, outside the search budget.
    """
    metadata = dict(value=result.value, cpu_seconds=result.cpu_seconds,
                    config=asdict(result.config), provenance=asdict(result.provenance),
                    progress=[asdict(row) for row in result.progress],
                    failures=[asdict(row) for row in result.failures],
                    has_controller=result.controller is not None)
    payload = json.dumps(metadata, allow_nan=False,
                         default=lambda value: value.item())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('wb') as stream:
            controller = result.controller
            if controller is None:
                np.savez_compressed(stream, metadata=np.array(payload))
            else:
                np.savez_compressed(
                    stream, metadata=np.array(payload), actions=controller.actions,
                    next_memory=controller.next_memory,
                    initial=np.array([controller.initial_action,
                                      controller.initial_next_memory], dtype=np.int64),
                )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_result(path: str | Path) -> SearchResult:
    """Load a result archive without executing serialized Python objects."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata'].item()))
        controller = None
        if metadata['has_controller']:
            first = archive['initial']
            controller = Controller(archive['actions'].copy(), archive['next_memory'].copy(),
                                    int(first[0]), int(first[1]))
    config = metadata['config']
    config['em'] = EMConfig(**config['em'])
    return SearchResult(controller, metadata['value'], metadata['cpu_seconds'],
                        [SearchProgress(**row) for row in metadata['progress']],
                        SearchConfig(**config), RunProvenance(**metadata['provenance']),
                        [SearchFailure(**row) for row in metadata['failures']])
