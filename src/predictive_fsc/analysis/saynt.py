"""Offline native-policy replay and storage attribution, outside solver execution."""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Literal, TypedDict


class PolicyAnalysis(TypedDict):
    value_source: Literal['solver_reported']
    replayed_value: float | None
    policy_bytes: int | None
    verification: Literal['unavailable', 'controller_replay', 'value_mismatch']


def _read_paynt_policy(text):
    """PAYNT writes A(observation,memory) and M(observation,memory) assignments."""
    actions = {}
    writes = {}
    memory_sizes = {}
    for assignment in text.strip().split(', '):
        if not assignment:
            continue
        variable, value = assignment.split('=')
        observation, memory = map(int, variable[2:-1].split(','))
        key = observation, memory
        memory_sizes[observation] = max(memory_sizes.get(observation, 1), memory + 1)
        if variable.startswith('A('):
            action_name = value.strip("{}'()")  # {'(a2)'} and {'a2'} both name action 2.
            actions[key] = int(action_name.removeprefix('a'))
        else:
            writes[key] = int(value)
    return actions, writes, memory_sizes


def _read_storm_policy(text, metadata):
    """Read node labels, deterministic actions and observation routes from DOT."""
    if any(marker in text for marker in ('cutoff', 'finite_mem', 'sched_')):
        raise ValueError('Storm graph omits its referenced cutoff policies')
    labels = {}
    actions = {}
    edges = []
    for line in text.splitlines():
        if ' -> ' in line:
            source, _, target = line.split()[:3]
            if source.startswith('"'):
                # "3c" -> 7 [ label= "0.5" ]; describes an observation branch.
                node = int(source.strip('"').removesuffix('c'))
                probability = float(line.split('label= "')[1].split('"')[0])
                edges.append((node, int(target), probability))
            else:
                # 3 -> "3c" [label= "{(a2)}"]; selects an action.
                label = line.split('label= "')[1].split('"')[0]
                if label.startswith('{(a'):
                    action = label.removeprefix('{(a').removesuffix(')}')
                    actions[int(source)] = int(action)
        elif '[ label = "' in line:
            node = int(line.split()[0])
            labels[node] = line.split('{')[1].split('}')[0].split(', ')

    initial = next(node for node, flags in labels.items() if 'init' in flags)
    node_observations = {}
    stage_counts = Counter()
    for node, flags in labels.items():
        for flag in flags:
            if flag.startswith('obs_'):
                observation = int(flag.removeprefix('obs_'))
                node_observations[node] = observation
                stage = metadata['observations'][observation][0]
                stage_counts[stage] += 1

    routes = {}
    for node, target, probability in edges:
        if probability <= 0 or target not in node_observations:
            continue
        key = node, node_observations[target]
        if key in routes and routes[key] != target:
            raise ValueError('ambiguous Storm observation routing')
        routes[key] = target
    size = max(stage_counts.values(), default=1)
    return actions, routes, initial, size


def replay_controller(model, metadata, text):
    """Replay a native controller on original rewards and terminal semantics.

    PAYNT memory writes happen with the action; Storm nodes advance after the
    next observation. Keep that timing explicit in the two branches below.
    """
    observation_ids = {tuple(pair): index for index, pair in enumerate(metadata['observations'])}
    storm = text.startswith('digraph')
    if storm:
        actions, routes, initial, size = _read_storm_policy(text, metadata)
        storage = 8 * len(actions) + 12 * len(routes)
    else:
        actions, writes, memory_sizes = _read_paynt_policy(text)
        initial = 0
        size = max(memory_sizes.values(), default=1)
        storage = 12 * (len(actions) + len(writes))
    storage += 8 * len(observation_ids)

    terminal_states = set(model['terminal_states'])
    terminal_actions = set(model['terminal_actions'])
    occupancy = {}
    for state, probability in enumerate(model['initial_belief']):
        if probability and state not in terminal_states:
            occupancy[state, initial, -1] = probability

    value = 0.
    for stage in range(model['horizon']):
        following = defaultdict(float)
        for (state, memory, observation), probability in occupancy.items():
            if storm:
                action = actions[memory]
            else:
                decision = observation_ids[stage, observation], memory
                action = actions[decision] if len(model['observation']) > 1 else 0
                write = writes.get(decision, 0)
            value += probability * model['reward'][state][action]
            if action in terminal_actions or stage + 1 == model['horizon']:
                continue

            targets = model['transition_destinations'][state][action]
            probabilities = model['transition_probabilities'][state][action]
            for target, transition_probability in zip(targets, probabilities):
                if target in terminal_states:
                    continue
                for next_observation, likelihood in enumerate(model['observation'][action][target]):
                    mass = probability * transition_probability * likelihood
                    if not mass:
                        continue
                    next_id = observation_ids[stage + 1, next_observation]
                    if storm:
                        next_memory = routes[memory, next_id]
                    else:
                        next_memory = write
                        if next_memory >= memory_sizes.get(next_id, 1):
                            next_memory = 0
                    following[target, next_memory, next_observation] += mass
        occupancy = following
    return value, size, storage


def analyze_saynt(directory, result) -> PolicyAnalysis:
    """Only attribute storage to an on-time export matching the reported return."""
    analysis: PolicyAnalysis = {
        'value_source': 'solver_reported',
        'replayed_value': None,
        'policy_bytes': None,
        'verification': 'unavailable',
    }
    if not result['policy'] or result['policy_cpu'] > result['config']['budget_seconds']:
        return analysis
    directory = Path(directory)
    model = json.loads((directory / 'model.json').read_text())
    metadata = json.loads((directory / 'conversion.json').read_text())
    try:
        text = (directory / result['policy']).read_text()
        ending = '}\n\n' if text.startswith('digraph') else '\n'
        if not text.endswith(ending):
            return analysis
        value, _, storage = replay_controller(model, metadata, text)
    except (ValueError, KeyError, StopIteration, TypeError, IndexError, OSError):
        # Native exporters can leave partial files or omit referenced policies.
        return analysis
    if math.isclose(value, result['value'], abs_tol=1e-7, rel_tol=1e-9):
        analysis.update(replayed_value=value, policy_bytes=storage, verification='controller_replay')
    else:
        analysis['verification'] = 'value_mismatch'
    return analysis
