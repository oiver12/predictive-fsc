"""Navigation controller occupancy and path sampling, independent of rendering."""

import numpy as np
from pathlib import Path

from ..dynamics import Dynamics
from ..io import load_problem
from .records import MapoRecord


def navigation_data(row: MapoRecord, models: Path):
    """Return grid width, exact pre-goal occupation and eight fixed-seed paths."""
    loaded = load_problem(models / (row['problem'] + '.toml'))
    problem = loaded.problem
    controller = row['saved'].controller
    if controller is None:
        return None
    model = problem.model
    # Navigation models use row-major state indices and neutral sN labels.
    width = int(problem.name.split('_')[1].removeprefix('W'))
    cells = [(s // width, s % width) for s in range(width * width)] + [None]
    # 1. Exact occupancy: exclude the absorbing tail and normalize pre-goal mass.
    active = np.array([cell is not None for cell in cells])
    dynamics = Dynamics(model, active, np.ones(model.num_actions, dtype=bool))
    mass = dynamics.initial_mass(controller.initial_action,controller.initial_next_memory,controller.num_memory)
    occupancy = model.initial_belief * active
    for t in range(problem.horizon - 1):
        mass[~active] = 0
        occupancy += mass.sum(axis=(1,2))
        joint = mass @ model.observation_probabilities.transpose(1,0,2)
        mass = dynamics.forward(joint,controller.actions[t],controller.next_memory[t])
    occupancy /= occupancy.sum()
    # 2. Eight sample paths from this controller, with a fixed random seed.
    rng = np.random.default_rng(0)
    paths = []
    for repeat in range(8):
        state = int(rng.choice(model.num_states,p=model.initial_belief))
        memory, observation = 0, -1
        for t in range(problem.horizon):
            if not active[state]:
                break
            cell = cells[state]
            assert cell is not None  # Active states are grid cells.
            paths.append(dict(path=repeat,stage=t,row=cell[0],column=cell[1]))
            if t == 0:
                action = controller.initial_action
                write = controller.initial_next_memory
            else:
                action = controller.actions[t - 1, memory, observation]
                write = controller.next_memory[t - 1, memory, observation]
            edges = model.transitions[action].getrow(state)
            state = int(rng.choice(edges.indices,p=edges.data/edges.data.sum()))
            observation = int(rng.choice(model.num_observations,p=model.observation_probabilities[action,state]))
            memory = write
    occupation_rows = []
    for cell, probability in zip(cells, occupancy):
        if cell is not None:
            occupation_rows.append(dict(row=cell[0], column=cell[1], probability=float(probability)))
    return width, occupation_rows, paths
