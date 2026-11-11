"""Generate grid navigation .pomdp + TOML files from explicit layout parameters.

Coordinates are zero-based (row,column). Movements slip to the perpendicular
sides. Beacon observations are noisy Manhattan distances plus a goal signal.
Entering the goal goes to a zero-reward absorbing state. Empty terminal lists
retain post-absorption observations in full-horizon fitting data, as in the paper.
"""
import argparse
from itertools import product
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from predictive_fsc import LoadedProblem, POMDP, Problem, save_problem
from generate_rocksample import coordinate  # Shared row,column argument parser.


def distance_distribution(distance: int, range_k: int, noise: float) -> dict:
    probabilities = {value:0.0 for value in [*range(range_k+1),None]}
    if distance <= range_k:
        probabilities[distance] = 1-noise
        targets = [d for d in (distance-1,distance+1) if 0 <= d <= range_k]
        if distance == range_k:
            targets.append(None)
        if not targets:
            probabilities[distance] = 1.0
    else:
        probabilities[None] = 1-noise
        targets = [range_k] if range_k == 0 else [range_k,range_k-1]
    for target in targets:
        probabilities[target] += noise/len(targets)
    return probabilities


def generate(size: int, start: tuple[int,int], goal: tuple[int,int],
             beacons: list[tuple[int,int]], range_k: int, horizon: int, *,
             hazards: list[tuple[int,int]] | None = None, p_intended: float = 0.8,
             p_side: float = 0.1, distance_noise: float = 0.05,
             goal_noise: float = 0.01, step_cost: float = -1.0,
             hazard_cost: float = -10.0, goal_reward: float = 100.0) -> LoadedProblem:
    start,goal = tuple(start),tuple(goal)
    beacons = [tuple(cell) for cell in beacons]
    hazards = [tuple(cell) for cell in (hazards or [])]
    if size < 2 or range_k < 0 or horizon < 1 or not beacons:
        raise ValueError('need size >= 2, range >= 0, positive horizon and at least one beacon')
    if any(not (0 <= r < size and 0 <= c < size) for r,c in [start,goal,*beacons,*hazards]):
        raise ValueError('all coordinates must be inside the grid')
    if len(set(hazards)) != len(hazards) or len(set(beacons)) != len(beacons):
        raise ValueError('hazards and beacons must not contain duplicates')
    if not np.isfinite([p_intended,p_side,distance_noise,goal_noise]).all() or min(p_intended,p_side,distance_noise,goal_noise) < 0 or max(distance_noise,goal_noise) > 1:
        raise ValueError('invalid transition/sensor probabilities')
    if not np.isclose(p_intended+2*p_side,1,rtol=0,atol=1e-12):
        raise ValueError('p-intended + 2*p-side must equal one')
    if not np.isfinite([step_cost,hazard_cost,goal_reward]).all():
        raise ValueError('rewards must be finite')
    cells = [(r,c) for r in range(size) for c in range(size)]
    S = len(cells)+1
    absorbing = S-1
    actions = ['north','south','east','west']
    deltas = [(-1,0),(1,0),(0,1),(0,-1)]
    sides = [(3,2),(3,2),(0,1),(0,1)]
    def move(cell,action):
        r,c = cell; dr,dc = deltas[action]
        return (r+dr,c+dc) if 0 <= r+dr < size and 0 <= c+dc < size else cell
    transitions = []
    rewards = np.zeros((S,4))
    for a in range(4):
        data,indices,indptr = [],[],[0]
        branches = [(a,p_intended),*( (side,p_side) for side in sides[a])]
        for s,cell in enumerate([*cells,None]):
            row = {}
            if cell is None or cell == goal:
                row[absorbing] = 1.0
                rewards[s,a] = goal_reward if cell == goal else 0
            else:
                rewards[s,a] = step_cost+(hazard_cost if cell in hazards else 0)
                for direction,probability in branches:
                    destination = move(cell,direction)
                    n = absorbing if destination == goal else destination[0]*size+destination[1]
                    row[n] = row.get(n,0.0)+probability
                    if destination == goal:
                        rewards[s,a] += goal_reward*probability
            total = sum(row.values())
            for n,p in sorted(row.items()):
                if p:
                    indices.append(n);data.append(p/total)
            indptr.append(len(data))
        transitions.append(csr_matrix((data,indices,indptr),shape=(S,S)))
    values = [*range(range_k+1),None]
    outcomes = [(*distances,signal) for distances in product(values,repeat=len(beacons)) for signal in (0,1)]
    observation = np.zeros((4,S,len(outcomes)))
    for s,cell in enumerate(cells):
        channels = [distance_distribution(abs(cell[0]-b[0])+abs(cell[1]-b[1]),range_k,distance_noise) for b in beacons]
        row = np.zeros(len(outcomes))
        for o,outcome in enumerate(outcomes):
            probability = 1.0
            for channel,distance in zip(channels,outcome[:-1]):
                probability *= channel[distance]
            row[o] = probability*(1-goal_noise if outcome[-1]==int(cell==goal) else goal_noise)
        observation[:,s] = row/row.sum()
    observation[:,absorbing,outcomes.index((*([None]*len(beacons)),0))] = 1.0
    initial = np.zeros(S);initial[start[0]*size+start[1]] = 1
    model = POMDP(transitions,observation,rewards,initial)
    state_names = [f'r{r}_c{c}' for r,c in cells]+['absorbing']
    observation_names = ['_'.join([*(f'd{d}' if d is not None else 'out' for d in row[:-1]),f'g{row[-1]}']) for row in outcomes]
    return LoadedProblem(Problem('navigation',model,horizon),state_names,actions,observation_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--size',type=int,required=True)
    parser.add_argument('--start',type=coordinate,required=True)
    parser.add_argument('--goal',type=coordinate,required=True)
    parser.add_argument('--beacons',type=coordinate,nargs='+',required=True)
    parser.add_argument('--hazards',type=coordinate,nargs='*',default=[])
    parser.add_argument('--range',dest='range_k',type=int,required=True)
    parser.add_argument('--horizon',type=int,required=True)
    parser.add_argument('--p-intended',type=float,default=0.8)
    parser.add_argument('--p-side',type=float,default=0.1)
    parser.add_argument('--distance-noise',type=float,default=0.05)
    parser.add_argument('--goal-noise',type=float,default=0.01)
    parser.add_argument('--step-cost',type=float,default=-1.0)
    parser.add_argument('--hazard-cost',type=float,default=-10.0)
    parser.add_argument('--goal-reward',type=float,default=100.0)
    parser.add_argument('--output',type=Path,required=True,help='output .toml path; writes a matching .pomdp')
    args = vars(parser.parse_args());output = args.pop('output')
    try:
        loaded = generate(**args)
        output.parent.mkdir(parents=True,exist_ok=True)
        save_problem(loaded,output)
    except (ValueError,OSError) as error:
        parser.exit(2,f'error: {error}\n')
    print(f'Wrote {output} and {output.with_suffix(".pomdp")}')


if __name__ == '__main__':
    main()
