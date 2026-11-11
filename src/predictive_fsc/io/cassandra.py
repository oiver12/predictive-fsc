"""Read standard Cassandra .pomdp models for the companion TOML loader.

Rewards are reduced to expected r(s,a); transition/observation-dependent reward
randomness is not retained. The source discount is ignored: the companion TOML
selects the undiscounted finite-horizon objective.
"""
from itertools import product
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from ..types import LoadedProblem, POMDP, Problem


def read_pomdp(path: str | Path, *, horizon: int, terminal_actions: list[str],
               terminal_states: list[str]) -> LoadedProblem:
    """Parse a .pomdp file and resolve the terminal labels."""
    if type(horizon) is not int or horizon < 1:
        raise ValueError('horizon must be a positive integer')
    path = Path(path)
    text = path.read_text()
    lines = [line.split('#',1)[0].strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    header, statements = {}, []
    i = 0
    while i < len(lines):
        if ':' not in lines[i]:
            raise ValueError(f'expected Cassandra directive: {lines[i]}')
        head, body = lines[i].split(':',1)
        i += 1
        values = [body.strip()]
        while i < len(lines) and ':' not in lines[i]:
            values.append(lines[i]); i += 1
        payload = ' '.join(values).strip()
        key = head.strip().lower()
        if key in ('discount','values','states','actions','observations'):
            if key in header:
                raise ValueError(f'duplicate Cassandra header: {key}')
            header[key] = payload
        else:
            statements.append((key,payload))
    if not all(key in header for key in ('discount','values','states','actions','observations')):
        raise ValueError('missing Cassandra preamble field')
    if header['values'] not in ('reward','cost'):
        raise ValueError('values must be reward or cost')
    spaces = []
    for key,prefix in (('states','s'),('actions','a'),('observations','o')):
        tokens = header[key].split()
        labels = [f'{prefix}{i}' for i in range(int(tokens[0]))] if len(tokens)==1 and tokens[0].isdigit() else tokens
        if not labels or len(set(labels)) != len(labels):
            raise ValueError(f'{key}: expected unique nonempty labels')
        spaces.append(labels)
    states, actions, observations = spaces
    S,A,O = map(len,spaces)

    def indices(token, labels):
        if token == '*':
            return list(range(len(labels)))
        if token.isdigit():
            index = int(token)
            if index < len(labels):
                return [index]
        elif token in labels:
            return [labels.index(token)]
        raise ValueError(f'unknown label/index: {token}')

    def table(tokens, rows, cols, *, identity=False):
        if tokens == ['uniform']:
            return np.full((rows,cols),1/cols)
        if tokens == ['identity'] and identity and rows == cols:
            return np.eye(rows)
        data = np.asarray(tokens,dtype=np.float64)
        if data.size != rows*cols or not np.isfinite(data).all():
            raise ValueError(f'expected {rows*cols} finite table entries')
        return data.reshape(rows,cols)

    T = [[{} for _ in states] for _ in actions]
    Z = np.zeros((A,S,O))
    initial = np.full(S,1/S)
    reward_rules = {}
    for key, payload in statements:
        if key in ('start','start include','start exclude'):
            tokens = payload.split()
            if key != 'start':
                chosen = [indices(token,states)[0] for token in tokens]
                initial = np.ones(S) if key == 'start exclude' else np.zeros(S)
                initial[chosen] = 0 if key == 'start exclude' else 1
                if initial.sum() == 0:
                    raise ValueError('empty starting state set')
                initial /= initial.sum()
            elif tokens == ['uniform']:
                initial = np.full(S,1/S)
            elif len(tokens) == 1 and (tokens[0] in states or tokens[0].isdigit()):
                initial = np.zeros(S); initial[indices(tokens[0],states)[0]] = 1
            else:
                initial = table(tokens,1,S)[0]
            continue
        if key not in ('t','o','r'):
            raise ValueError(f'unsupported Cassandra directive: {key}')
        fields = [field.split() for field in payload.split(':')]
        if any(not field for field in fields):
            raise ValueError(f'empty Cassandra field: {payload}')
        ai = indices(fields[0][0],actions)
        if key in ('t','o'):
            cols = S if key == 't' else O
            if len(fields) == 1:
                block = table(fields[0][1:],S,cols,identity=True)
                si = list(range(S))
            elif len(fields) == 2:
                si = indices(fields[1][0],states)
                if fields[1][1:] == ['reset'] and key == 't':
                    block = np.tile(initial,(len(si),1))
                else:
                    row = table(fields[1][1:],1,cols)[0]
                    block = np.tile(row,(len(si),1))
            elif len(fields) == 3:
                si = indices(fields[1][0],states)
                successors = indices(fields[2][0],states if key=='t' else observations)
                value = table(fields[2][1:],1,1).item()
                for a,s,n in product(ai,si,successors):
                    if key == 't':
                        if value: T[a][s][n] = value
                        else: T[a][s].pop(n,None)
                    else:
                        Z[a,s,n] = value
                continue
            else:
                raise ValueError('invalid T/O directive')
            for a in ai:
                for row,s in enumerate(si):
                    if key == 't':
                        T[a][s] = {int(n):float(block[row,n]) for n in np.flatnonzero(block[row])}
                    else:
                        Z[a,s] = block[row]
            continue
        if not 2 <= len(fields) <= 4:
            raise ValueError('invalid reward directive')
        si = indices(fields[1][0],states)
        if len(fields) == 4:
            ni = indices(fields[2][0],states)
            oi = indices(fields[3][0],observations)
            values = table(fields[3][1:],1,1).item()
        elif len(fields) == 3:
            ni = indices(fields[2][0],states)
            oi = list(range(O))
            values = table(fields[2][1:],1,O)
        else:
            ni,oi = list(range(S)),list(range(O))
            values = table(fields[1][1:],S,O)
        for a,s in product(ai,si):
            reward_rules.setdefault((a,s),[]).append((ni,oi,values))
    transitions = []
    rewards = np.zeros((S,A))
    for a in range(A):
        data, cols, pointers = [],[],[0]
        for s in range(S):
            row = T[a][s]
            # Only one S-by-O reward workspace, never an A*S*S*O tensor.
            full = np.zeros((S,O))
            for ni,oi,values in reward_rules.get((a,s),[]):
                full[np.ix_(ni,oi)] = values
            for n,probability in sorted(row.items()):
                cols.append(n); data.append(probability)
                rewards[s,a] += probability * (Z[a,n] @ full[n])
            pointers.append(len(data))
        transitions.append(csr_matrix((data,cols,pointers),shape=(S,S)))
    if header['values'] == 'cost':
        rewards *= -1
    action_terminals = [actions.index(label) for label in terminal_actions]
    state_terminals = [states.index(label) for label in terminal_states]
    model = POMDP(transitions,Z,rewards,initial,action_terminals,state_terminals)
    name = path.stem
    loaded = LoadedProblem(Problem(name,model,horizon),*spaces)
    validate_problem(loaded)
    # Normalize the initial belief (not T or Z) to absorb rounding in the source file.
    loaded.problem.model.initial_belief /= loaded.problem.model.initial_belief.sum()
    return loaded


def validate_problem(loaded: LoadedProblem) -> None:
    """Validate parsed external data once; numerical dataclasses remain plain."""
    model = loaded.problem.model
    S,A,O = len(loaded.states),len(loaded.actions),len(loaded.observations)
    for array,shape,name in ((model.rewards,(S,A),'rewards'),
            (model.observation_probabilities,(A,S,O),'observation probabilities'),
            (model.initial_belief,(S,),'initial belief')):
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f'{name}: expected finite array of shape {shape}')
    for array in (model.initial_belief,model.observation_probabilities):
        if np.any(array < 0) or not np.allclose(array.sum(axis=-1),1,rtol=1e-5,atol=1e-8):
            raise ValueError('probabilities must be nonnegative and sum to one')
    for matrix in model.transitions:
        if matrix.shape != (S,S) or not np.isfinite(matrix.data).all() or np.any(matrix.data < 0) or not np.allclose(np.asarray(matrix.sum(axis=1)).ravel(),1,rtol=1e-5,atol=1e-8):
            raise ValueError('transition probabilities must be finite, nonnegative and sum to one')
    for terminals,labels in ((model.terminal_actions,loaded.actions),(model.terminal_states,loaded.states)):
        if len(set(terminals)) != len(terminals) or any(i < 0 or i >= len(labels) for i in terminals):
            raise ValueError('terminal indices must be unique and in range')
