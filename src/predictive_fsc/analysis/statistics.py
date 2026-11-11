"""Numerical summaries and the paper's comparison conventions.

These functions operate on records in memory, without plotting, file access or
solver calls. Keep the paper's exclusions, weighting and CPU comparisons here.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar

import numpy as np

from ..types import SearchResult

if TYPE_CHECKING:
    from .records import HalvingAllocation, MapoRecord, RunRecord


Row = TypeVar('Row')
Key = TypeVar('Key')
Trace: TypeAlias = list[tuple[float, float]]  # (charged seconds, incumbent return)


def grouped(rows: Iterable[Row], key: Callable[[Row], Key]) -> dict[Key, list[Row]]:
    """Group with a typed key function, retaining row order within each group."""
    groups: dict[Key, list[Row]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return groups


def value_at(trace: Trace, seconds: float) -> float | None:
    """Last saved incumbent available at this CPU time, or None before the first."""
    index = bisect_right(trace, seconds, key=lambda point: point[0]) - 1
    if index < 0:
        return None
    return trace[index][1]


def incumbents(trace: Trace) -> Trace:
    """Keep only improvements so large random-start studies stay cheap to plot."""
    improvements = []
    for seconds, value in sorted(trace):
        if not improvements or value > improvements[-1][1]:
            improvements.append((seconds, value))
    return improvements


def weighted_iqm(values, weights):
    """Average the weighted empirical quantile mass [.25, .75]."""
    values = np.asarray(values)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    weights = weights[order] / weights.sum()
    quantile_end = np.cumsum(weights)
    quantile_start = quantile_end - weights
    # Retain fractional weights at the two trim boundaries. Do not clip scores.
    retained = np.minimum(quantile_end, .75) - np.maximum(quantile_start, .25)
    retained = np.maximum(retained, 0)
    return float(2 * (values[order] @ retained))


def halving_trace(arms: Sequence[SearchResult], allocation: HalvingAllocation) -> Trace:
    """Map each arm's CPU time to total search CPU, excluding late candidates."""
    budget = allocation['arm_budget_cpu_seconds']
    survivors = allocation['survivors']
    half_budget = budget / 2
    first_phase_cpu = len(arms) * half_budget
    trace = []
    for arm in arms:
        cutoff = budget if arm.config.memory in survivors else half_budget
        for candidate in arm.progress:
            if candidate.cpu_seconds > cutoff:
                continue
            if candidate.cpu_seconds <= half_budget:
                total_cpu = len(arms) * candidate.cpu_seconds
            else:
                continued_cpu = candidate.cpu_seconds - half_budget
                total_cpu = first_phase_cpu + len(survivors) * continued_cpu
            trace.append((total_cpu, candidate.value))
    return sorted(trace)


def table(rows: Sequence[RunRecord]):
    """One summary per benchmark/method; count missing values without filling zeros."""
    output = []
    primary = (row for row in rows if not row['arm'])
    for (problem, method), group in sorted(grouped(primary, lambda row: (row['problem'], row['method'])).items()):
        values = [row['value'] for row in group if row['value'] is not None]
        sizes = [row['policy_bytes'] for row in group if row['policy_bytes'] is not None]
        replayed = [row['analysis']['replayed_value'] for row in group
                    if row['solver'] == 'saynt' and row['analysis']['replayed_value'] is not None]
        default_source = 'initial_vector' if method.startswith('fivi') else 'policy_iteration'
        memories = [str(row['memory']) for row in sorted(group, key=lambda row: row['seed'])]
        first = group[0]
        output.append({
            'problem': problem, 'method': method, 'runs': len(group), 'available': len(values),
            'value_source': first['analysis']['value_source'] if first['solver'] == 'saynt' else default_source,
            'replayed_available': len(replayed),
            'replayed_mean': float(np.mean(replayed)) if replayed else None,
            'mean': float(np.mean(values)) if values else None,
            'sd': float(np.std(values, ddof=1)) if len(values) > 1 else None,
            'policy_bytes': float(np.mean(sizes)) if sizes else None,
            'memories': ','.join(memories),
        })
    return output


def anytime_data(rows: Sequence[RunRecord], navigation: bool = False):
    """Summarize eligible incumbents; start only when every seed has a value."""
    eligible = (row for row in rows if not row['arm'] and row['trace']
                and not row['method'].endswith('_extended'))
    data = []
    for (problem, method), group in sorted(grouped(eligible, lambda row: (row['problem'], row['method'])).items()):
        traces = [row['trace'] for row in group]
        start = max(trace[0][0] for trace in traces)
        times = {seconds for trace in traces for seconds, _ in trace if seconds >= start}
        times.update(row['duration'] for row in group if row['duration'] >= start)
        for seconds in sorted(times):
            values = []
            for trace in traces:
                value = value_at(trace, seconds)
                assert value is not None  # The shared start is after every first incumbent.
                values.append(value)
            center = float(np.median(values) if navigation else np.mean(values))
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else None
            data.append({
                'problem': problem, 'method': method, 'cpu_seconds': seconds,
                'available': len(traces),
                'median': center if navigation else None,
                'mean': None if navigation else center,
                'sd': sd,
            })
    return data


def _initialization_anchors(rows: Sequence[RunRecord]) -> dict[str, tuple[float, float]]:
    anchors = {}
    for problem, group in sorted(grouped(rows, lambda row: row['problem']).items()):
        reactive = [row['value'] for row in group if row['method'] == 'reactive' and row['value'] is not None]
        values = [row['value'] for row in group if row['value'] is not None]
        if reactive and max(values) > max(reactive):
            anchors[problem] = max(reactive), max(values)
    return anchors


def _normalized_iqm(values: Iterable[tuple[str, float | None]],
                    anchors: dict[str, tuple[float, float]]) -> dict[str, int | float] | None:
    """Normalize returns, then give each benchmark equal total seed weight."""
    available = [(problem, value) for problem, value in values if value is not None]
    if not available:
        return None
    counts = Counter(problem for problem, _ in available)
    scores = []
    weights = []
    for problem, value in available:
        reactive, best_known = anchors[problem]
        scores.append((value - reactive) / (best_known - reactive))
        weights.append(1 / counts[problem])
    return {'runs': len(available), 'benchmarks': len(counts), 'iqm': weighted_iqm(scores, weights)}


def _initialization_curves(runs: Sequence[MapoRecord], anchors: dict[str, tuple[float, float]]):
    """Compare M4 on the same (benchmark, seed) cohort for both methods."""
    m4 = [row for row in runs if row['memory'] == 4 and row['trace']]
    cohorts = defaultdict(set)
    for row in m4:
        cohorts[row['method']].add((row['problem'], row['seed']))
    common = cohorts['predictive'] & cohorts['random']
    paired = [row for row in m4 if (row['problem'], row['seed']) in common]
    if not paired:
        return []

    # Begin once every run has a controller; stopped arms retain their incumbent.
    start = max(row['trace'][0][0] for row in paired)
    times = {150} | {seconds for row in paired for seconds, _ in row['trace']
                     if start <= seconds <= 150}
    curves = []
    for method in ('predictive', 'random'):
        group = [row for row in paired if row['method'] == method]
        for seconds in sorted(times):
            if seconds < start:
                continue
            values = [(row['problem'], value_at(row['trace'], seconds)) for row in group]
            summary = _normalized_iqm(values, anchors)
            assert summary is not None  # Every paired run has an incumbent by start.
            curves.append({'method': method, 'cpu_seconds': seconds, **summary})
    return curves


def initialization_data(rows: Sequence[RunRecord]):
    """Paper protocol: first candidate, equal 75-second CPU, paired M4 IQM."""
    # These showed no meaningful memory benefit in the paper; keep its fixed
    # exclusion set when normalizing initialization scores for new runs.
    excluded = {'4x4', 'bridge_repair', 'line4_2goals'}
    anchors = _initialization_anchors(rows)
    # Use each arm, excluding the additional selected-result file.
    runs = [row for row in rows
            if row['solver'] == 'mapo' and row['problem'] in anchors and row['problem'] not in excluded
            and row['method'] in ('predictive', 'random') and row['memory'] in (2, 4, 8, 16)
            and not row['halving_selected']]

    comparisons: list[dict[str, Any]] = []
    for row in runs:
        first_value = next((candidate.value for candidate in row['candidates']
                            if candidate.origin == row['method']), None)
        values = [('first', first_value), ('75_seconds', value_at(row['trace'], 75))]
        for panel, value in values:
            comparisons.append({
                'panel': panel, 'problem': row['problem'], 'method': row['method'],
                'memory': row['memory'], 'seed': row['seed'], 'value': value,
            })

    summaries = []
    for (panel, method, memory), group in sorted(grouped(
        comparisons, lambda row: (row['panel'], row['method'], row['memory'])
    ).items()):
        summary = _normalized_iqm(((row['problem'], row['value']) for row in group), anchors)
        if summary is not None:
            summaries.append({'panel': panel, 'method': method, 'memory': memory, **summary})
    summaries.sort(key=lambda row: (row['panel'] != 'first', row['method'], row['memory']))
    curves = _initialization_curves(runs, anchors)

    anchor_rows = []
    for problem, (reactive, best_known) in anchors.items():
        anchor_rows.append(dict(problem=problem, reactive=reactive, best_known=best_known,
                                included=problem not in excluded))
    comparisons.sort(key=lambda row: row['panel'] != 'first')
    return anchor_rows, comparisons, summaries, curves
