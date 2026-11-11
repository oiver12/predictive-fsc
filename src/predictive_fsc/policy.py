"""Finite-horizon deterministic controller evaluation and policy iteration.

The compact state is (environment state, memory, previous action). The previous
action determines the current observation law. No augmented transition tensor
is formed. Public controllers retain separate action and memory-write tables.
"""

import hashlib

import numpy as np

from numba import njit
from .dynamics import Dynamics
from .types import Controller, PolicyIterationResult, Problem

TIE_TOLERANCE = 1e-12
REACHABLE_TOLERANCE = 1e-15


def _check_controller(problem: Problem, controller: Controller) -> None:
    model = problem.model
    if problem.horizon < 1 or controller.actions.ndim != 3:
        raise ValueError(
            "expected a positive horizon and three-dimensional controller tables"
        )
    expected = (problem.horizon - 1, controller.num_memory, model.num_observations)
    if (
        controller.num_memory < 1
        or controller.actions.shape != expected
        or controller.next_memory.shape != expected
    ):
        raise ValueError(
            "controller tables must have shape (H-1,M,O) matching the problem"
        )
    for table, size in (
        (controller.actions, model.num_actions),
        (controller.next_memory, controller.num_memory),
    ):
        if table.dtype.kind not in "iu" or np.any(table < 0) or np.any(table >= size):
            raise ValueError("controller tables must contain in-range integer indices")
    for value, size in (
        (controller.initial_action, model.num_actions),
        (controller.initial_next_memory, controller.num_memory),
    ):
        if not isinstance(value, (int, np.integer)) or not 0 <= value < size:
            raise ValueError("initial controller entries must be in-range integers")


def _masks(model):
    active = np.ones(model.num_states, dtype=bool)
    active[model.terminal_states] = False
    continuing = np.ones(model.num_actions, dtype=bool)
    continuing[model.terminal_actions] = False
    return active, continuing


def _stage_value(model, q, actions, writes):
    chosen = q[:, actions.T, writes.T]  # (S,O,M)
    return (model.observation_probabilities.transpose(1, 0, 2) @ chosen).transpose(
        0, 2, 1
    )


def _backward(model, controller, dynamics):
    following = np.zeros((model.num_states, controller.num_memory))
    targets = []
    for index in reversed(range(controller.horizon - 1)):
        q = dynamics.q_values(following)
        targets.append(q)
        following = _stage_value(
            model, q, controller.actions[index], controller.next_memory[index]
        )
    return dynamics.q_values(following), list(reversed(targets))


def evaluate_controller(problem: Problem, controller: Controller) -> float:
    """Exact expected undiscounted return; terminal states have zero continuation."""
    _check_controller(problem, controller)
    model = problem.model
    active, continuing = _masks(model)
    dynamics = Dynamics(model, active, continuing)
    # Evaluation needs only one continuation buffer, not a full horizon trace.
    following = np.zeros((model.num_states, controller.num_memory))
    for index in reversed(range(controller.horizon - 1)):
        q = dynamics.q_values(following)
        following = _stage_value(
            model, q, controller.actions[index], controller.next_memory[index]
        )
    q = dynamics.q_values(following)
    return float(
        model.initial_belief
        @ q[:, controller.initial_action, controller.initial_next_memory]
    )


def _posterior(model, mass):
    """State beliefs for improving each (memory, observation) decision.

    Unused cells borrow the state belief given only their observation. These
    fallback weights enable improvement; only the unchanged joint mass is
    propagated through the environment.
    """
    joint = mass @ model.observation_probabilities.transpose(1, 0, 2)  # (S,M,O)
    weights = joint.sum(axis=0)  # (M,O)
    reachable = weights > REACHABLE_TOLERANCE
    posterior = joint / np.where(reachable, weights, 1.0)[None]
    posterior *= reachable[None]
    posterior = posterior.transpose(1, 2, 0)  # (M,O,S): one state belief per cell

    if not reachable.all():
        # Ignore memory to obtain P(state | observation) for unused cells.
        observation_joint = joint.sum(axis=1)  # (S,O)
        observation_weights = observation_joint.sum(axis=0)
        seen = observation_weights > REACHABLE_TOLERANCE
        observation_posterior = (
            observation_joint / np.where(seen, observation_weights, 1.0)[None]
        )
        memory, observation = np.nonzero(~reachable & seen[None])
        posterior[memory, observation] = observation_posterior[:, observation].T
        weights[memory, observation] = observation_weights[observation]

    # Flatten only at the matrix-product boundary; rows are (memory, observation).
    return posterior.reshape(-1, model.num_states), weights, joint


def _improve_initial(model, controller, q):
    M = controller.num_memory
    scores = model.initial_belief @ q.reshape(model.num_states, -1)
    current = controller.initial_action * M + controller.initial_next_memory
    best = scores.max()
    if scores[current] >= best - TIE_TOLERANCE:
        return False
    chosen = int(np.argmax(scores >= best - TIE_TOLERANCE))
    controller.initial_action, controller.initial_next_memory = divmod(chosen, M)
    return True


@njit(cache=True)
def _greedy_improve_row(scores, weights, actions, writes):
    """Greedy improvement loop over separate action/write tables."""
    M, O = actions.shape
    changed = False
    for m in range(M):
        for o in range(O):
            row = m * O + o
            if weights[row] <= REACHABLE_TOLERANCE:
                continue
            best = scores[row, 0]
            for column in range(1, scores.shape[1]):
                if scores[row, column] > best:  # noqa: PLR1730 - explicit compare in Numba loop
                    best = scores[row, column]
            current = actions[m, o] * M + writes[m, o]
            if scores[row, current] >= best - TIE_TOLERANCE:
                continue
            for column in range(scores.shape[1]):
                if scores[row, column] >= best - TIE_TOLERANCE:
                    actions[m, o], writes[m, o] = column // M, column % M
                    changed = True
                    break
    return changed


def _improve_stage(model, controller, index, q, posterior, weights):
    scores = posterior @ q.reshape(model.num_states, -1)
    return _greedy_improve_row(
        scores,
        weights.ravel(),
        controller.actions[index],
        controller.next_memory[index],
    )


def _controller_key(controller):
    digest = hashlib.sha256(controller.actions.tobytes())
    digest.update(controller.next_memory.tobytes())
    return controller.initial_action, controller.initial_next_memory, digest.digest()


def policy_iteration(
    problem: Problem, initial: Controller, *, max_sweeps: int = 100
) -> PolicyIterationResult:
    """Alternating forward/backward stage updates; input controller is unchanged.

    Stop after a complete sweep makes no changes. Stable ties keep the incumbent
    within 1e-12; otherwise choose the first (action,next-memory) in index order.
    Cycles and capped results have converged=False; cycled identifies a repeated
    full-sweep controller. Neither is silently called final.
    """
    _check_controller(problem, initial)
    if type(max_sweeps) is not int or max_sweeps < 0:
        raise ValueError("max_sweeps must be a nonnegative integer")
    model = problem.model
    controller = Controller(
        initial.actions.astype(np.int64, order="C", copy=True),
        initial.next_memory.astype(np.int64, order="C", copy=True),
        initial.initial_action,
        initial.initial_next_memory,
    )
    active, continuing = _masks(model)
    dynamics = Dynamics(model, active, continuing)
    q0, targets = _backward(model, controller, dynamics)
    value = float(
        model.initial_belief
        @ q0[:, controller.initial_action, controller.initial_next_memory]
    )
    values = [value]
    seen = {_controller_key(controller)}
    for sweep in range(max_sweeps):
        # Forward: update decisions and propagate their actual probability mass.
        changed = _improve_initial(model, controller, q0)
        mass = dynamics.initial_mass(
            controller.initial_action,
            controller.initial_next_memory,
            controller.num_memory,
        )
        posteriors = []
        for index in range(controller.horizon - 1):
            posterior, weights, joint = _posterior(model, mass)
            posteriors.append((posterior, weights))
            if _improve_stage(
                model, controller, index, targets[index], posterior, weights
            ):
                changed = True
            if index + 1 < controller.horizon - 1:
                mass = dynamics.forward(
                    joint, controller.actions[index], controller.next_memory[index]
                )
        # Backward: update decisions using fresh continuation values.
        following = np.zeros((model.num_states, controller.num_memory))
        for index in reversed(range(controller.horizon - 1)):
            posterior, weights = posteriors.pop()
            q = dynamics.q_values(following)
            targets[index] = q
            if _improve_stage(model, controller, index, q, posterior, weights):
                changed = True
            following = _stage_value(
                model, q, controller.actions[index], controller.next_memory[index]
            )
        q0 = dynamics.q_values(following)
        if _improve_initial(model, controller, q0):
            changed = True
        value = float(
            model.initial_belief
            @ q0[:, controller.initial_action, controller.initial_next_memory]
        )
        values.append(value)
        if not changed:
            return PolicyIterationResult(controller, value, sweep + 1, True, values)
        key = _controller_key(controller)
        if key in seen:
            return PolicyIterationResult(controller, value, sweep + 1, False, values, cycled=True)
        seen.add(key)
    return PolicyIterationResult(controller, value, max_sweeps, False, values)


def random_controller(
    problem: Problem, memory: int, rng: np.random.Generator
) -> Controller:
    """Independent uniform actions/writes; memory=1 is the reactive initializer."""
    if type(memory) is not int or memory < 1:
        raise ValueError("memory must be a positive integer")
    # Draw flat (action,next-memory) indices in the paper's order.
    initial = rng.integers(problem.model.num_actions * memory, size=memory)
    stages = rng.integers(
        problem.model.num_actions * memory,
        size=(problem.horizon - 1, problem.model.num_observations, memory),
    ).transpose(0, 2, 1)
    return Controller(
        np.ascontiguousarray(stages // memory),
        np.ascontiguousarray(stages % memory),
        int(initial[0] // memory), int(initial[0] % memory),
    )
