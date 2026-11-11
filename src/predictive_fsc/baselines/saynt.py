"""Finite-horizon SAYNT conversion and the existing local engine image."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Literal, TypedDict, cast
import uuid

import numpy as np

from . import _docker, fivi
from ..types import Problem


@dataclass(frozen=True)
class SayntConfig:
    budget_seconds: float
    seed: int = 0
    paynt_seconds: int = 60
    storm_seconds: int = 10

    def __post_init__(self) -> None:
        if not math.isfinite(self.budget_seconds) or self.budget_seconds <= 0:
            raise ValueError('SAYNT budget_seconds must be finite and positive')
        if min(self.paynt_seconds, self.storm_seconds) < 1:
            raise ValueError('SAYNT phases must be positive integer seconds')


class ConversionMetadata(TypedDict):
    horizon: int
    reward_shift: float
    states: list[tuple[int, int, int]]
    observations: list[tuple[int, int]]


class SayntProgress(TypedDict):
    controller: str
    value: float
    solver_seconds: float
    size: int
    charged_cpu: float


class SayntResult(TypedDict):
    solver: Literal['saynt']
    problem: str
    config: dict[str, Any]  # Serialized settings at the JSON boundary.
    image_id: str
    model_sha256: str
    wall_seconds: float
    status: str
    returncode: int | None
    value: float | None
    size: int | None
    policy: str | None
    policy_cpu: float | None
    trace: list[SayntProgress]
    charged_cpu: float
    charged_wall: float
    value_source: Literal['solver_reported']


def finite_horizon_drn(problem: Problem) -> tuple[str, ConversionMetadata]:
    model, horizon = problem.model, problem.horizon
    terminal = set(model.terminal_states)
    # Keep padding positive too: upstream cannot unfold an all-zero reward model.
    shift = max(1., 1. - float(model.rewards.min()))
    pairs = [(int(s), int(o)) for s, o in zip(*np.nonzero((model.observation_probabilities > 0).any(axis=0))) if s not in terminal]
    observations = [(0, -1)] + [(t, o) for t in range(1, horizon) for o in range(model.num_observations)]
    observations += [(t, -2) for t in range(1, horizon)] + [(horizon, -3)]
    states = [(0, -1, -1)] + [(t, s, o) for t in range(1, horizon) for s, o in pairs]
    states += [(t, -2, -2) for t in range(1, horizon)] + [(horizon, -3, -3)]
    present = {(t, o) for t, _, o in states}
    observations = [pair for pair in observations if pair in present]
    obs_ids = {pair: i for i, pair in enumerate(observations)}
    ids = {key: i for i, key in enumerate(states)}
    done = len(states) - 1
    active_belief = model.initial_belief.copy()
    active_belief[model.terminal_states] = 0
    initial = [np.asarray(matrix.T @ active_belief).ravel() for matrix in model.transitions]
    lines = ['@type: POMDP', '@parameters', '', '@reward_models', 'reward', '@nr_states', str(len(states)),
             '@nr_choices', str((len(states) - 1) * model.num_actions + 1), '@model']
    for index, (t, state, observation) in enumerate(states):
        lines.append(f'state {index} {{{obs_ids[t, observation]}}} [0]' + (' init' if index == 0 else '') + (' done' if index == done else ''))
        for action in range(1 if index == done else model.num_actions):
            if state >= 0:
                reward = float(model.rewards[state, action])
            elif state == -1:
                reward = float(active_belief @ model.rewards[:, action])
            else:
                reward = 0.
            reward = 0. if state == -3 else reward + shift
            lines.append(f'\taction (a{action}) [{reward:.17g}]')
            if t >= horizon - 1:
                successors = {done: 1.}
            elif state == -2 or action in model.terminal_actions:
                successors = {ids[t + 1, -2, -2]: 1.}
            else:
                if state == -1:
                    distribution = enumerate(initial[action])
                    inactive = 1. - float(active_belief.sum())
                else:
                    matrix = model.transitions[action]
                    begin, end = matrix.indptr[state:state + 2]
                    distribution = zip(matrix.indices[begin:end], matrix.data[begin:end])
                    inactive = 0.
                successors = {}
                for target, probability in distribution:
                    if target in terminal:
                        inactive += probability
                    else:
                        for obs, likelihood in enumerate(model.observation_probabilities[action, target]):
                            if probability * likelihood > 0:
                                key = ids[t + 1, int(target), obs]
                                successors[key] = successors.get(key, 0.) + float(probability * likelihood)
                if inactive > 0:
                    successors[ids[t + 1, -2, -2]] = inactive
            lines.extend(f'\t\t{target} : {p:.17g}' for target, p in successors.items())
    return '\n'.join(lines) + '\n', ConversionMetadata(
        horizon=horizon, reward_shift=shift, states=states, observations=observations)


def load_result(directory: str | Path) -> SayntResult:
    return cast(SayntResult, json.loads((Path(directory) / 'result.json').read_text()))


def run(problem: Problem, config: SayntConfig, output_dir: str | Path, image: str) -> SayntResult:
    wall, cpu = time.monotonic(), time.process_time()
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / 'result.json').unlink(missing_ok=True)
    drn, metadata = finite_horizon_drn(problem)
    (root / 'sketch.drn').write_text(drn)
    (root / 'sketch.props').write_text('R{"reward"}max=? [F "done"]\n')
    (root / 'conversion.json').write_text(json.dumps(metadata))
    model = dict(fivi.export_model(problem), terminal_actions=problem.model.terminal_actions,
                 terminal_states=problem.model.terminal_states)
    (root / 'model.json').write_text(json.dumps(model, allow_nan=False))
    settings = dict(asdict(config), setup_cpu_seconds=time.process_time() - cpu)
    (root / 'settings.json').write_text(json.dumps(settings))
    name = f'predictive-saynt-{uuid.uuid4().hex}'
    worker = Path(__file__).with_name('_saynt_worker.py').resolve()
    command = [*_docker.run_command(name, '/tmp:rw,exec,size=512m'), '--cpus', '1',
               '--mount', f'type=bind,src={root},dst=/work',
               '--mount', f'type=bind,src={worker},dst=/worker.py,readonly',
               '-e', f'PYTHONHASHSEED={config.seed % 2**32}', '--entrypoint', 'python', image, '/worker.py', '/work']
    with _docker.cleanup(name):
        with (root / 'worker.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=10 * config.budget_seconds + 90)
    result = cast(SayntResult, json.loads((root / 'worker-result.json').read_text()))
    if result['status'] in ('error', 'wall_guard'):
        raise ValueError(f'SAYNT failed; see {root / "solver.log"}')
    result.update(solver='saynt', problem=problem.name, config=asdict(config), image_id=image,
                  model_sha256=hashlib.sha256(drn.encode()).hexdigest(), wall_seconds=time.monotonic() - wall)
    _docker.write_result(root / 'result.json', result)
    return result
