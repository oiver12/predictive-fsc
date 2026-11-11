"""Load a problem description (.toml) and its standard Cassandra model (.pomdp)."""
from pathlib import Path
from collections import Counter
import json
import re
import tomllib

import numpy as np

from ..dynamics import canonical
from ..types import LoadedProblem
from .cassandra import read_pomdp


def load_problem(path: str | Path) -> LoadedProblem:
    """Resolve the model relative to its descriptor; retain labels for reporting."""
    path = Path(path)
    with path.open('rb') as file:
        description = tomllib.load(file)
    required = {'model','horizon','discount','terminal_actions','terminal_states'}
    if description.keys() != required:
        raise ValueError(f'problem description must contain exactly {sorted(required)}')
    if not isinstance(description['model'],str) or not description['model']:
        raise ValueError('model must be a nonempty path')
    if type(description['horizon']) is not int or description['horizon'] < 1:
        raise ValueError('horizon must be a positive integer')
    if type(description['discount']) not in (int,float) or description['discount'] != 1:
        raise ValueError('only undiscounted finite-horizon problems (discount=1.0) are supported')
    for field in ('terminal_actions','terminal_states'):
        labels = description[field]
        if not isinstance(labels,list) or any(not isinstance(label,str) for label in labels) or len(set(labels)) != len(labels):
            raise ValueError(f'{field} must be a list of unique names')
    loaded = read_pomdp(path.parent/description['model'],horizon=description['horizon'],
        terminal_actions=description['terminal_actions'],terminal_states=description['terminal_states'])
    loaded.problem.name = path.stem
    return loaded


def save_problem(loaded: LoadedProblem, path: str | Path) -> None:
    """Write a .toml description and a same-stem standard .pomdp model."""
    path = Path(path)
    if path.suffix != '.toml':
        raise ValueError('save_problem expects a .toml destination')
    model_path = path.with_suffix('.pomdp')
    write_pomdp(loaded,model_path)
    model = loaded.problem.model
    actions = [loaded.actions[i] for i in model.terminal_actions]
    states = [loaded.states[i] for i in model.terminal_states]
    # Cassandra mnemonic names contain no quotes or special TOML characters.
    def names(values):
        return '['+', '.join(f'"{name}"' for name in values)+']'
    model_name = json.dumps(model_path.name,ensure_ascii=False)  # TOML basic-string escaping.
    path.write_text(f'model = {model_name}\nhorizon = {loaded.problem.horizon}\n'
                    f'discount = 1.0\nterminal_actions = {names(actions)}\nterminal_states = {names(states)}\n')


def write_pomdp(loaded: LoadedProblem, path: str | Path) -> None:
    """Export arrays as standard T/O/R declarations without changing sparse T/Z."""
    for labels in (loaded.states,loaded.actions,loaded.observations):
        if not labels or len(set(labels)) != len(labels) or any(re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*',name) is None for name in labels):
            raise ValueError('Cassandra names must be unique identifiers starting with a letter')
    model = loaded.problem.model
    def numbers(values):
        return ' '.join(repr(float(v)) for v in values)
    lines = ['# Finite-horizon settings and termination are in the companion TOML file.',
             'discount: 1.0','values: reward','states: '+' '.join(loaded.states),
             'actions: '+' '.join(loaded.actions),'observations: '+' '.join(loaded.observations),
             ]
    if np.count_nonzero(model.initial_belief) == 1 and model.initial_belief.max() == 1:
        lines.append('start: '+loaded.states[int(model.initial_belief.argmax())])
    elif np.array_equal(model.initial_belief,np.full(model.num_states,1/model.num_states)):
        lines.append('start: uniform')
    else:
        lines.append('start: '+numbers(model.initial_belief))

    for a,action in enumerate(loaded.actions):
        matrix = canonical(model.transitions[a])
        rows = []
        for state in range(model.num_states):
            start,end = matrix.indptr[state:state+2]
            rows.append(dict(zip(matrix.indices[start:end],matrix.data[start:end])))
        identical = all(row == rows[0] for row in rows)
        identity_cost = 1+sum(sum(row.get(n,0) != (1 if n==state else 0)
                                for n in row.keys() | {state})
                            for state,row in enumerate(rows))
        use_identity = not identical and identity_cost < matrix.nnz
        if identical:
            for n,probability in rows[0].items():
                lines.append(f'T: {action} : * : {loaded.states[n]} {float(probability)!r}')
        else:
            if use_identity:
                lines.append(f'T: {action} identity')
            for state,row in enumerate(rows):
                baseline = {state:1.0} if use_identity else {}
                for n in sorted(row.keys() | baseline.keys()):
                    probability = row.get(n,0.0)
                    if probability != baseline.get(n,0.0):
                        lines.append(f'T: {action} : {loaded.states[state]} : {loaded.states[n]} {float(probability)!r}')

        # Expected r(s,a) has already been reduced from source rewards. Preserve
        # it when standard readers integrate R over rounded T/Z row sums.
        rewards = []
        for state,row in enumerate(rows):
            weight = sum(probability*model.observation_probabilities[a,n].sum()
                         for n,probability in row.items())
            rewards.append(float(model.rewards[state,a]/weight))
        default,frequency = Counter(rewards).most_common(1)[0]
        use_default = default != 0 and frequency > rewards.count(0.0)+1
        if use_default:
            lines.append(f'R: {action} : * : * : * {default!r}')
        for state,reward in enumerate(rewards):
            if reward != (default if use_default else 0.0):
                lines.append(f'R: {action} : {loaded.states[state]} : * : * {reward!r}')

    observations = model.observation_probabilities
    groups = Counter(table.tobytes() for table in observations)
    common,count = groups.most_common(1)[0]
    if count > 1:
        index = next(i for i,table in enumerate(observations) if table.tobytes()==common)
        tables = [('*',observations[index])]+[(action,table) for action,table in zip(loaded.actions,observations) if table.tobytes()!=common]
    else:
        tables = list(zip(loaded.actions,observations))
    for action,table in tables:
        if np.array_equal(table,np.full(table.shape,1/model.num_observations)):
            lines.append(f'O: {action} uniform')
            continue
        rows = [tuple(row) for row in table]
        default,count = Counter(rows).most_common(1)[0]
        if count > 1:
            lines.append(f'O: {action} : * '+numbers(default))
            for state,row in enumerate(rows):
                if row != default:
                    lines.append(f'O: {action} : {loaded.states[state]} '+numbers(row))
        else:
            lines.append(f'O: {action}')
            lines.extend(numbers(row) for row in table)
    Path(path).write_text('\n'.join(lines)+'\n')
