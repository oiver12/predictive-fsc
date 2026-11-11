"""Normalize saved MAPO/FiVI/SAYNT runs and identify halving arms and selections.

The record definitions below describe what records() returns. Halving selections
are explicit so summaries need no filesystem probes. TypedDict describes the
loaded dictionaries to type checkers; it does not validate external JSON.
"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict, cast

from ..baselines import fivi, saynt
from ..io import load_result
from ..types import SearchProgress, SearchResult
from .saynt import PolicyAnalysis, analyze_saynt
from .statistics import Trace, halving_trace, incumbents


class CommonRecord(TypedDict):
    problem: str
    method: str
    seed: int
    memory: int | None
    value: float | None
    policy_bytes: int | None
    trace: Trace
    duration: float
    source: Path
    arm: bool
    halving_selected: bool


class MapoRecord(CommonRecord):
    solver: Literal['mapo']
    candidates: list[SearchProgress]
    cutoff: float | None
    saved: SearchResult


class FiviRecord(CommonRecord):
    solver: Literal['fivi']
    upper_bound: float | None


class SayntRecord(CommonRecord):
    solver: Literal['saynt']
    analysis: PolicyAnalysis


RunRecord: TypeAlias = MapoRecord | FiviRecord | SayntRecord


class HalvingAllocation(TypedDict):
    arm_budget_cpu_seconds: float
    survivors: list[int]
    memory: int | None
    arm_results: dict[str, str]


# The extended-reference plans (main-extended.toml, navigation-extended.toml) use 7,200 s.
EXTENDED_SECONDS = 7200


def _mapo_record(path: Path, *, arm: bool = False) -> MapoRecord:
    result = load_result(path)
    cutoff = result.config.cpu_seconds
    candidates = result.eligible

    storage = None
    if result.controller is not None:
        storage = result.controller.actions.nbytes + result.controller.next_memory.nbytes + 16
    return {
        'solver': 'mapo',
        'problem': result.provenance.build_identity['problem_name'],
        'method': 'reactive' if result.config.memory == 1 else result.config.initialization,
        'seed': result.config.seed,
        'memory': result.config.memory,
        'value': result.value,
        'policy_bytes': storage,
        'trace': incumbents([(row.cpu_seconds, row.value) for row in candidates]),
        'duration': cutoff if cutoff is not None else result.cpu_seconds,
        'candidates': candidates,
        'cutoff': cutoff,
        'source': path,
        'arm': arm,
        'halving_selected': False,
        'saved': result,
    }


def _halving_records(path: Path, allocation: HalvingAllocation) -> list[MapoRecord]:
    selected = _mapo_record(path)
    selected['halving_selected'] = True
    arms: list[MapoRecord] = []
    for filename in allocation['arm_results'].values():
        arm = _mapo_record(path.parent / filename, arm=True)
        cutoff = allocation['arm_budget_cpu_seconds'] / 2
        if arm['memory'] in allocation['survivors']:
            cutoff = allocation['arm_budget_cpu_seconds']
        arm['candidates'] = [row for row in arm['candidates'] if row.cpu_seconds <= cutoff]
        arm['trace'] = incumbents([(row.cpu_seconds, row.value) for row in arm['candidates']])
        arm['cutoff'] = cutoff
        arms.append(arm)
    selected['trace'] = incumbents(halving_trace([arm['saved'] for arm in arms], allocation))
    selected['duration'] = (len(arms) + len(allocation['survivors'])) * allocation['arm_budget_cpu_seconds'] / 2
    selected['memory'] = allocation['memory']
    return [*arms, selected]


def _baseline_record(path: Path) -> FiviRecord | SayntRecord:
    result = json.loads(path.read_text())
    analysis: PolicyAnalysis | None = None
    if result['solver'] == 'fivi':
        result = fivi.load_result(path.parent)
        clock = 'charged_' + result['config']['budget_clock']
        trace = [(row[clock], row['lower_bound']) for row in result['trace']]
        model = json.loads((path.parent / 'model.json').read_text())
        counts = result['vectors_per_stage']
        memory = max(counts) if counts else None
        storage = sum(counts) * (8 * len(model['initial_belief']) + 4) if counts else None
        method = 'fivi_extended' if result['config']['budget_seconds'] == EXTENDED_SECONDS else 'fivi'
    elif result['solver'] == 'saynt':
        result = saynt.load_result(path.parent)
        trace = incumbents([(row['charged_cpu'], row['value']) for row in result['trace']])
        analysis = analyze_saynt(path.parent, result)
        memory = result['size']
        storage = analysis['policy_bytes']
        method = 'saynt_extended' if result['config']['budget_seconds'] == EXTENDED_SECONDS else 'saynt'
    else:
        raise ValueError(f"unknown saved solver: {result['solver']}")
    record: CommonRecord = {
        'problem': result['problem'], 'method': method, 'seed': result['config']['seed'],
        'memory': memory, 'value': result['value'], 'policy_bytes': storage,
        'trace': trace, 'duration': result['config']['budget_seconds'],
        'source': path, 'arm': False, 'halving_selected': False,
    }
    if result['solver'] == 'saynt':
        assert analysis is not None
        return {'solver': 'saynt', **record, 'analysis': analysis}
    return {'solver': 'fivi', **record, 'upper_bound': result['upper_bound']}


def records(paths: Iterable[str | Path]) -> list[RunRecord]:
    """Load primary results and their arms once, even when a directory lists both."""
    files: set[Path] = set()
    for path in map(Path, paths):
        if path.is_dir():
            files.update(path.glob('*.npz'))
            files.update(path.rglob('result.json'))
        else:
            files.add(path)

    allocations: dict[Path, HalvingAllocation] = {}
    arm_files: set[Path] = set()
    for path in files:
        allocation_path = path.with_suffix('.halving.json')
        if path.suffix == '.npz' and allocation_path.exists():
            allocation = cast(HalvingAllocation, json.loads(allocation_path.read_text()))
            allocations[path] = allocation
            for filename in allocation['arm_results'].values():
                arm_files.add(path.parent / filename)

    loaded: list[RunRecord] = []
    for path in sorted(files - arm_files):
        if path in allocations:
            loaded.extend(_halving_records(path, allocations[path]))
        elif path.suffix == '.npz':
            loaded.append(_mapo_record(path))
        else:
            loaded.append(_baseline_record(path))
    return loaded
