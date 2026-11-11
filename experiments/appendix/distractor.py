"""Optional appendix study: cue-aware sampling and the distractor-side T-maze.

Runs the fixed paper study (8 K values x 30 repetitions); --plot reads saved results only.
"""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from predictive_fsc import (Controller, EMConfig, load_problem, sample_trajectories,
                            fit_em, extract_controller, evaluate_controller, policy_iteration)


def cue_controller(loaded):
    """Model indices: forward/left/right; four cue combinations, three corridors, junction."""
    problem = loaded.problem
    shape = problem.horizon - 1, 2, problem.model.num_observations
    actions = np.zeros(shape, dtype=np.int64)
    writes = np.broadcast_to(np.arange(2)[None, :, None], shape).copy()
    writes[:, :, :2] = 0
    writes[:, :, 2:4] = 1
    actions[:, 0, 7] = 1
    actions[:, 1, 7] = 2
    return Controller(actions, writes, 0, 0)


def _information(joint):
    total = joint.sum()
    if not total:
        return None
    p = joint / total
    independent = p.sum(axis=1, keepdims=True) @ p.sum(axis=0, keepdims=True)
    nonzero = p > 0
    return float(np.sum(p[nonzero] * np.log2(p[nonzero] / independent[nonzero])))


def diagnostic(problem, controller):
    """Exact memory information at first junction arrival, before taking its action.

    State order: cue, side, position (-1 through junction), as exported.
    These models have deterministic action-independent observations.
    """
    model = problem.model
    positions = model.num_states // 4
    mass = np.zeros((model.num_states, 2))
    mass[:, 0] = model.initial_belief
    joint = np.zeros((2, 2, 2))
    observation = model.observation_probabilities[0].argmax(axis=1)
    for t in range(problem.horizon):
        for block in range(4):
            s = (block + 1) * positions - 1
            joint[block // 2, block % 2] += mass[s]
            mass[s] = 0
        following = np.zeros_like(mass)
        for state, memory in zip(*np.nonzero(mass)):
            action = controller.initial_action if t == 0 else controller.actions[t - 1, memory, observation[state]]
            write = controller.initial_next_memory if t == 0 else controller.next_memory[t - 1, memory, observation[state]]
            if action not in model.terminal_actions:
                row = model.transitions[action].getrow(state)
                following[row.indices, write] += mass[state, memory] * row.data
        mass = following
    cue, side = _information(joint.sum(axis=1)), _information(joint.sum(axis=0))
    label = ('none' if cue is None or max(cue, side) < .05 else
             'mixed' if abs(cue - side) <= .05 else 'cue' if cue > side else 'side')
    return dict(mi_memory_cue=cue, mi_memory_side=side, encoded_bit=label,
                reach_junction_mass=float(joint.sum()))



def run(loaded, directory, *, seed=0, trajectories=4000, fits=4, em_iterations=100):
    problem = loaded.problem
    rng = np.random.default_rng(seed)
    batch = sample_trajectories(problem, cue_controller(loaded), trajectories, rng, exploration=.1)
    candidates = []
    for fit in range(fits):
        fit_seed = int(rng.integers(2**32))
        fitted = fit_em(problem, batch, 2, seed=fit_seed, config=EMConfig(max_iterations=em_iterations))
        controller = extract_controller(problem, fitted.model)
        row = dict(fit=fit, seed=fit_seed, seed_value=evaluate_controller(problem, controller),
                   **diagnostic(problem, controller))
        refined = policy_iteration(problem, controller, max_sweeps=300)
        row.update(value=refined.value, sweeps=refined.sweeps)
        candidates.append(row)
    result = dict(problem=problem.name, seed=seed, trajectories=trajectories, fits=fits,
                  em_iterations=em_iterations, exploration=.1, max_sweeps=300,
                  value=max(row['value'] for row in candidates), candidates=candidates)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def plot(directory):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    root = Path(directory)
    groups = defaultdict(list)
    for path in sorted(root.glob('D*/seed*/result.json')):
        result = json.loads(path.read_text())
        groups[int(result['problem'].rsplit('D', 1)[1])].append(result)
    data = []
    for k, group in sorted(groups.items()):
        counts = Counter(c['encoded_bit'] for result in group for c in result['candidates'])
        values = [result['value'] for result in group]
        data.append(dict(k=k, repetitions=len(group), median=float(np.median(values)),
                         q25=float(np.quantile(values, .25)), q75=float(np.quantile(values, .75)),
                         **{label: counts[label] / sum(counts.values()) for label in ('cue','side','mixed','none')}))
    with (root / 'distractor.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['k','repetitions','median','q25','q75','cue','side','mixed','none'])
        writer.writeheader()
        writer.writerows(data)
    figure, axes = plt.subplots(1, 2, figsize=(9, 3), layout='constrained')
    x, bottom = np.arange(len(data)), np.zeros(len(data))
    for label in ('cue','side','mixed','none'):
        fraction = np.array([row[label] for row in data])
        axes[0].bar(x, fraction, bottom=bottom, label=label)
        bottom += fraction
    axes[0].legend(fontsize='small')
    axes[0].set(ylabel='Fraction of extracted seeds')
    axes[1].plot(x, [row['median'] for row in data], marker='o')
    axes[1].fill_between(x, [row['q25'] for row in data], [row['q75'] for row in data], alpha=.2)
    axes[1].set(ylabel='Best refined return per repetition')
    for ax in axes:
        ax.set(xticks=x, xticklabels=[row['k'] for row in data], xlabel='Distractor markers K')
    for suffix in ('.svg', '.pdf'):
        figure.savefig(root / ('distractor' + suffix))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/appendix-distractor'))
    parser.add_argument('--plot', action='store_true', help='plot existing results without fitting')
    args = parser.parse_args()
    if args.plot:
        plot(args.output)
        return
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    models = Path(__file__).resolve().parents[2] / 'models'
    for k in (0, 1, 2, 4, 8, 16, 32, 64):
        loaded = load_problem(models / f'tmaze_side_D{k}.toml')
        for seed in range(30):
            root = args.output / f'D{k}' / f'seed{seed}'
            if not (root / 'result.json').exists():
                run(loaded, root, seed=seed)


if __name__ == '__main__':
    main()
