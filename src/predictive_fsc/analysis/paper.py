"""Draw paper panels and export CSV tables from saved-result analysis.

render() is the figures --paper entry point. Loading, replay and numerical
summaries live in sibling modules; this module owns presentation and output files.
"""

import csv
from collections.abc import Sequence
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from .navigation import navigation_data
from .records import FiviRecord, MapoRecord, RunRecord, records
from .statistics import anytime_data, grouped, initialization_data, table


def write_csv(path, rows):
    rows = list(rows)
    columns = list(dict.fromkeys(column for row in rows for column in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _save(figure, output, name):
    paths = []
    for suffix in ('.svg', '.pdf'):
        path = output / (name + suffix)
        figure.savefig(path)
        paths.append(path)
    plt.close(figure)
    return paths


def anytime(rows: Sequence[RunRecord], output: Path, navigation: bool = False) -> list[Path]:
    """Draw mean/sample SD for benchmarks or median for navigation."""
    problems = sorted({row['problem'] for row in rows})
    data = anytime_data(rows, navigation)
    figure, axes = plt.subplots(len(problems), 1, squeeze=False, layout='constrained',
                                figsize=(8, max(3, 2.6 * len(problems))))
    for problem, ax in zip(problems, axes[:, 0]):
        for method, group in sorted(grouped(
            (row for row in data if row['problem'] == problem), lambda row: row['method']
        ).items()):
            times = [row['cpu_seconds'] for row in group]
            centers = np.asarray([row['median' if navigation else 'mean'] for row in group])
            deviations = np.asarray([row['sd'] or 0 for row in group])
            line, = ax.step(times, centers, where='post', label=method)
            if not navigation:
                ax.fill_between(times, centers - deviations, centers + deviations,
                                step='post', alpha=.15, color=line.get_color())
        ax.set(title=problem, xlabel='Total search CPU seconds', ylabel='Expected return')
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize='small')
    files = [write_csv(output / 'anytime.csv', data)]
    files.extend(_save(figure, output, 'anytime'))
    return files


def initialization(rows: Sequence[RunRecord], output: Path) -> list[Path]:
    """Draw and export the paper's initialization comparisons."""
    anchor_rows, comparisons, summaries, curves = initialization_data(rows)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3), layout='constrained')
    for method in ('predictive', 'random'):
        for ax, panel in zip(axes[:2], ('first', '75_seconds')):
            group = [row for row in summaries if row['method'] == method and row['panel'] == panel]
            ax.plot([row['memory'] for row in group], [row['iqm'] for row in group], marker='o', label=method)
            ax.set(xlabel='Memory cells', ylabel='IQM memory benefit', title=panel.replace('_', ' '))
        group = [row for row in curves if row['method'] == method]
        axes[2].step([row['cpu_seconds'] for row in group], [row['iqm'] for row in group], where='post', label=method)
    axes[2].set(xlabel='CPU seconds per M4 run', ylabel='IQM memory benefit', title='M4 anytime')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize='small')

    files = [write_csv(output / 'anchors.csv', anchor_rows)]
    files.append(write_csv(output / 'initialization-runs.csv', comparisons))
    files.append(write_csv(output / 'initialization.csv', summaries))
    files.append(write_csv(output / 'initialization-anytime.csv', curves))
    files.extend(_save(figure, output, 'initialization'))
    return files


def navigation_behavior(row: MapoRecord, models: Path, output: Path) -> list[Path]:
    data = navigation_data(row, models)
    if data is None:
        return []
    width, occupation_rows, paths = data
    grid = np.zeros((width, width))
    for cell in occupation_rows:
        grid[cell['row'], cell['column']] = cell['probability']
    figure, axes = plt.subplots(1, 2, figsize=(8, 4), layout='constrained')
    image = axes[0].imshow(grid,origin='upper')
    figure.colorbar(image,ax=axes[0],label='Normalized pre-goal occupation')
    for repeat in range(8):
        group = [p for p in paths if p['path'] == repeat]
        axes[1].plot([p['column'] for p in group],[p['row'] for p in group],alpha=.6)
    axes[1].set(xlim=(-.5,width-.5),ylim=(width-.5,-.5),title='Eight paths; sampling seed 0')
    for ax in axes:
        ax.set(xlabel='Column',ylabel='Row')
    stem = row['problem']+'-behavior'
    files = [write_csv(output / (stem + '-occupancy.csv'), occupation_rows)]
    files.append(write_csv(output / (stem + '-paths.csv'), paths))
    files.extend(_save(figure, output, stem))
    return files


def _tiger(rows: Sequence[RunRecord], output: Path) -> list[Path]:
    values = []
    references: list[FiviRecord] = []
    for row in rows:
        if row['solver'] == 'mapo' and row['method'] == 'random':
            values.extend(candidate.value for candidate in row['candidates'] if candidate.origin == 'random')
        elif row['solver'] == 'fivi' and row['method'] == 'fivi' and row['value'] is not None:
            references.append(row)
    figure, ax = plt.subplots(figsize=(6, 3), layout='constrained')
    ax.hist(values, bins=30)
    if references:
        lower_bound = max(row['value'] for row in references if row['value'] is not None)
        ax.axvline(lower_bound, color='black', label='FiVI lower bound')
        ax.legend()
    ax.set(xlabel='Return after random initialization and PI', ylabel='Starts')
    starts = [dict(start=index, value=value) for index, value in enumerate(values)]
    bounds = [dict(lower=row['value'], upper=row['upper_bound']) for row in references]
    files = [write_csv(output / 'tiger-starts.csv', starts)]
    files.append(write_csv(output / 'tiger-reference.csv', bounds))
    files.extend(_save(figure, output, 'tiger'))
    return files


def render(kind, paths, output, models=Path('models')):
    """Entry point for `figures --paper`: load saved runs, summarize, plot, save."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = records(paths)
    if not rows:
        raise ValueError('no completed result files supplied')

    if kind == 'tiger':
        files = _tiger(rows, output)
    else:
        files = [write_csv(output / 'returns-and-storage.csv', table(rows))]
        files.extend(anytime(rows, output, navigation=kind == 'navigation'))
        if kind == 'main':
            files.extend(initialization(rows, output))
        else:
            candidates = (row for row in rows if row['solver'] == 'mapo'
                          and row['method'] == 'predictive' and row['value'] is not None)
            for _, group in sorted(grouped(candidates, lambda row: row['problem']).items()):
                # Highest-return seed; ties go to the smallest seed number.
                selected = max(group, key=lambda row: (
                    float('-inf') if row['value'] is None else row['value'], -row['seed']))
                files.extend(navigation_behavior(selected, Path(models), output))

    sources = [dict(problem=row['problem'], method=row['method'], seed=row['seed'], path=str(row['source']))
               for row in rows]
    files.append(write_csv(output / 'sources.csv', sources))
    return files
