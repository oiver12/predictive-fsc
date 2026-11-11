"""Sparse finite-horizon input and charged-time results for the corrected FiVI image."""

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

from ..dynamics import canonical
from ..types import Problem
from . import _docker


@dataclass(frozen=True)
class FiviConfig:
    budget_seconds: float
    seed: int = 0
    budget_clock: Literal['cpu', 'wall'] = 'cpu'
    parallelism: int = 1
    heap: str = '3g'
    iterations: int | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.budget_seconds) or self.budget_seconds <= 0:
            raise ValueError('FiVI budget_seconds must be finite and positive')
        if self.budget_clock not in ('cpu', 'wall'):
            raise ValueError('FiVI budget_clock must be cpu or wall')
        if self.parallelism < 1:
            raise ValueError('FiVI parallelism must be a positive integer')
        if self.iterations is not None and self.iterations < 1:
            raise ValueError('FiVI iterations must be a positive integer or omitted')


class ExportedModel(TypedDict):
    transition_destinations: list[list[list[int]]]
    transition_probabilities: list[list[list[float]]]
    observation: list[list[list[float]]]
    reward: list[list[float]]
    initial_belief: list[float]
    horizon: int


class FiviSweep(TypedDict):
    iteration: int
    lower_bound: float
    upper_bound: float
    charged_cpu: float
    charged_wall: float
    vectors_per_stage: list[int]


class FiviResult(TypedDict):
    """A saved FiVI run, shared by the runner and offline analysis."""

    status: str
    value: float | None
    upper_bound: float | None
    vectors_per_stage: list[int] | None
    iteration: int | None
    charged_cpu: float
    charged_wall: float
    trace: list[FiviSweep]
    solver: Literal['fivi']
    problem: str
    config: dict[str, Any]  # Serialized settings; FiviConfig checks them when loaded.
    image_id: str
    model_sha256: str
    wall_seconds: float
    verification: Literal['initial_vector', 'none']
    policy_replay: Literal[False]


def export_model(problem: Problem) -> ExportedModel:
    """Export sparse successors; encode declared termination as zero-reward tails.

    A terminal action keeps its immediate reward, then enters a sink. Declared
    terminal states earn no reward, including at initialization. Other absorbing
    states retain their ordinary rewards and observation laws.
    """
    model = problem.model
    S, A = model.num_states, model.num_actions
    sink = bool(model.terminal_actions or model.terminal_states)
    states = S + int(sink)
    destinations = [[[] for _ in range(A)] for _ in range(states)]
    probabilities = [[[] for _ in range(A)] for _ in range(states)]
    terminal_states, terminal_actions = set(model.terminal_states), set(model.terminal_actions)
    for action, source in enumerate(model.transitions):
        matrix = canonical(source)
        for state in range(states):
            if state == S or state in terminal_states or action in terminal_actions:
                destinations[state][action] = [S]
                probabilities[state][action] = [1.0]
            else:
                start, end = matrix.indptr[state:state + 2]
                destinations[state][action] = matrix.indices[start:end].tolist()
                probabilities[state][action] = matrix.data[start:end].tolist()
    observation = np.pad(model.observation_probabilities, ((0, 0), (0, int(sink)), (0, 0)))
    reward = np.pad(model.rewards, ((0, int(sink)), (0, 0)))
    belief = np.pad(model.initial_belief, (0, int(sink)))
    if sink:
        observation[:, S, 0] = 1
        reward[model.terminal_states, :] = 0
    return ExportedModel(transition_destinations=destinations, transition_probabilities=probabilities,
                observation=observation.tolist(), reward=reward.tolist(),
                initial_belief=belief.tolist(), horizon=problem.horizon)


def parse_output(path: str | Path, model: ExportedModel, config: FiviConfig) -> dict[str, Any]:
    """Check initial vectors and keep the last completed eligible sweep, never a late one."""
    belief = np.asarray(model['initial_belief'])
    trace: list[FiviSweep] = []
    done = None
    sweeps = 0
    with Path(path).open() as stream:
        for line in stream:
            message = json.loads(line)
            kind, row = message['kind'], message['value']
            if kind == 'bound':
                sweeps += 1
                vector = np.asarray(row['initial_vector'], dtype=float)
                value = float(vector @ belief)
                if not math.isclose(value, row['lower_bound'], abs_tol=1e-9, rel_tol=1e-9):
                    raise ValueError(f"FiVI initial-vector bound mismatch at iteration {row['iteration']}")
                if row[f'charged_{config.budget_clock}'] <= config.budget_seconds:
                    trace.append(cast(FiviSweep, {key: value for key, value in row.items() if key != 'initial_vector'}))
            elif kind == 'done':
                done = row
    if done is None or done['iteration'] != sweeps:
        raise ValueError('incomplete FiVI output: missing completion or sweep records')
    last = trace[-1] if trace else None
    used = done[f'charged_{config.budget_clock}']
    return dict(status='budget' if used > config.budget_seconds else done['status'],
                value=last['lower_bound'] if last else None,
                upper_bound=last['upper_bound'] if last else None,
                vectors_per_stage=last['vectors_per_stage'] if last else None,
                iteration=last['iteration'] if last else None,
                charged_cpu=done['charged_cpu'], charged_wall=done['charged_wall'], trace=trace)


def load_result(directory: str | Path) -> FiviResult:
    """Validate completed output before skipping an existing baseline job."""
    directory = Path(directory)
    result = cast(FiviResult, json.loads((directory / 'result.json').read_text()))
    data = (directory / 'model.json').read_bytes()
    if hashlib.sha256(data).hexdigest() != result['model_sha256']:
        raise ValueError('saved FiVI model checksum mismatch')
    parsed = parse_output(directory / 'stdout.jsonl', json.loads(data), FiviConfig(**result['config']))
    if any(result[key] != value for key, value in parsed.items()):
        raise ValueError('saved FiVI result disagrees with its solver output')
    return result


def run(problem: Problem, config: FiviConfig, output_dir: str | Path, image: str) -> FiviResult:
    """Run the existing image and save model, native output/log, and one JSON result.

    Java charges solver work and excludes startup/model parsing/reporting. The
    adapter's wall timeout is only a hang guard. No complete policy is exported.
    """
    started = time.monotonic()
    model = export_model(problem)
    data = (json.dumps(model, allow_nan=False, separators=(',', ':')) + '\n').encode()
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'result.json').unlink(missing_ok=True)
    (directory / 'model.json').write_bytes(data)
    name = f'predictive-fivi-{uuid.uuid4().hex}'
    command = [*_docker.run_command(name, '/tmp:rw,size=512m'), '--cpus', str(config.parallelism),
               '--mount', f'type=bind,src={directory},dst=/work,readonly',
               '-e', f'FIVI_HEAP={config.heap}', '-e', f'FIVI_PROCESSORS={config.parallelism}',
               '-e', 'JAVA_TOOL_OPTIONS=-ea -Xss8m -XX:+UseSerialGC', image,
               '--model', '/work/model.json', '--seed', str(config.seed),
               '--budget', str(config.budget_seconds), '--clock', config.budget_clock,
               '--parallelism', str(config.parallelism), '--iterations', str(config.iterations or 0)]
    with _docker.cleanup(name):
        with (directory / 'stdout.jsonl').open('w') as stdout, (directory / 'stderr.log').open('w') as stderr:
            subprocess.run(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                           timeout=2 * config.budget_seconds + 60, check=True)
    parsed = parse_output(directory / 'stdout.jsonl', model, config)
    result = FiviResult(solver='fivi', problem=problem.name, config=asdict(config),
                  image_id=image, model_sha256=hashlib.sha256(data).hexdigest(),
                  wall_seconds=time.monotonic() - started,
                  verification='initial_vector' if parsed['value'] is not None else 'none',
                  policy_replay=False, **parsed)
    _docker.write_result(directory / 'result.json', result)
    return result
