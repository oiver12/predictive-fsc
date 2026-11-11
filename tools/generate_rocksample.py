"""Generate a RockSample .pomdp model and TOML problem description.

Coordinates are zero-based (row,column). Rock order defines the bit positions.
An eastward exit is absorbing; empty terminal lists retain the paper's full-H
training trajectories.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from predictive_fsc import LoadedProblem, POMDP, Problem, save_problem


def generate(size: int, rocks: list[tuple[int,int]], horizon: int, *,
             start: tuple[int,int] | None = None, half_distance: float = 20.0,
             step_cost: float = -0.01, sample_good: float = 10.0,
             sample_bad: float = -10.0, exit_reward: float = 10.0) -> LoadedProblem:
    start = (size//2,0) if start is None else tuple(start)
    rocks = [tuple(cell) for cell in rocks]
    if size < 2 or horizon < 1 or not rocks or len(set(rocks)) != len(rocks):
        raise ValueError('need size >= 2, positive horizon and distinct rock positions')
    if any(not (0 <= r < size and 0 <= c < size) for r,c in [start,*rocks]):
        raise ValueError('start and rocks must be inside the grid')
    if not np.isfinite(half_distance) or half_distance <= 0:
        raise ValueError('half-distance must be finite and positive')
    if not np.isfinite([step_cost,sample_good,sample_bad,exit_reward]).all():
        raise ValueError('rewards must be finite')
    masks = 1 << len(rocks)
    cells = [(r,c,bits) for r in range(size) for c in range(size) for bits in range(masks)]
    S = len(cells)+1
    exit_state = S-1
    actions = ['north','south','east','west','sample']+[f'check_{i}' for i in range(len(rocks))]
    deltas = [(-1,0),(1,0),(0,1),(0,-1)]
    rock_at = {cell:i for i,cell in enumerate(rocks)}
    transitions = []
    rewards = np.full((S,len(actions)),step_cost,dtype=np.float64)
    rewards[exit_state] = 0
    observations = np.zeros((len(actions),S,3)); observations[:,:,0] = 1
    for a,action in enumerate(actions):
        successors = np.arange(S)
        for s,(r,c,bits) in enumerate(cells):
            if a < 4:
                dr,dc = deltas[a]
                nr,nc = r+dr,c+dc
                if action == 'east' and c == size-1:
                    successors[s] = exit_state
                    rewards[s,a] = exit_reward
                elif 0 <= nr < size and 0 <= nc < size:
                    successors[s] = (nr*size+nc)*masks+bits
            elif action == 'sample':
                rock = rock_at.get((r,c))
                good = rock is not None and bool(bits & (1 << rock))
                rewards[s,a] = sample_good if good else sample_bad
                if good:
                    successors[s] = (r*size+c)*masks+(bits & ~(1 << rock))
            else:
                rock = a-5
                distance = abs(r-rocks[rock][0])+abs(c-rocks[rock][1])
                correct = 0.5+0.5*2.0**(-distance/half_distance)
                p_good = correct if bits & (1 << rock) else 1-correct
                observations[a,s] = (0,p_good,1-p_good)
        transitions.append(csr_matrix((np.ones(S),(np.arange(S),successors)),shape=(S,S)))
    initial = np.zeros(S)
    offset = (start[0]*size+start[1])*masks
    initial[offset:offset+masks] = 1/masks
    model = POMDP(transitions,observations,rewards,initial)
    names = [f'r{r}_c{c}_b{bits}' for r,c,bits in cells]+['exit']
    return LoadedProblem(Problem('rocksample',model,horizon),names,actions,['null','good','bad'])


def coordinate(value: str) -> tuple[int,int]:
    try:
        row,column = value.split(',')
        return int(row),int(column)
    except ValueError as error:
        raise argparse.ArgumentTypeError('use row,column, e.g. 1,2') from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--size',type=int,required=True)
    parser.add_argument('--rocks',type=coordinate,nargs='+',required=True)
    parser.add_argument('--horizon',type=int,required=True)
    parser.add_argument('--start',type=coordinate,help='default: size//2,0')
    parser.add_argument('--half-distance',type=float,default=20.0)
    parser.add_argument('--step-cost',type=float,default=-0.01)
    parser.add_argument('--sample-good',type=float,default=10.0)
    parser.add_argument('--sample-bad',type=float,default=-10.0)
    parser.add_argument('--exit-reward',type=float,default=10.0)
    parser.add_argument('--output',type=Path,required=True,help='output .toml path; writes a matching .pomdp')
    args = parser.parse_args()
    try:
        loaded = generate(args.size,args.rocks,args.horizon,start=args.start,half_distance=args.half_distance,
                          step_cost=args.step_cost,sample_good=args.sample_good,sample_bad=args.sample_bad,
                          exit_reward=args.exit_reward)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        save_problem(loaded,args.output)
    except (ValueError,OSError) as error:
        parser.exit(2,f'error: {error}\n')
    print(f'Wrote {args.output} and {args.output.with_suffix(".pomdp")}')


if __name__ == '__main__':
    main()
