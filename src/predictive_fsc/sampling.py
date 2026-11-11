"""Numba rollout loop and cached row-local sampling CDFs."""

import numpy as np
from numba import njit

from .dynamics import canonical
from .policy import _check_controller, _masks
from .types import Controller, Problem, TrajectoryBatch


@njit(cache=True, nogil=True)
def _csr_choice_cdfs(data, indptr, totals):
    cdf = np.empty(data.size)
    for state in range(totals.size):
        start, end = indptr[state : state + 2]
        cumulative = 0.0
        for i in range(start, end):
            cumulative += data[i] / totals[state]
            cdf[i] = cumulative
        for i in range(start, end):
            cdf[i] /= cumulative
    return cdf


def _prepare_sampling(model):
    """Row-local CDFs, built once per model (models are not modified after loading)."""
    cache = model._cache
    if "sampling" not in cache:
        pointers, successors, cumulative = [], [], []
        offset = 0
        for source in model.transitions:
            matrix = canonical(source)
            totals = np.asarray(matrix.sum(axis=1)).ravel()
            cumulative.append(_csr_choice_cdfs(matrix.data, matrix.indptr, totals))
            pointers.append(matrix.indptr.astype(np.int64) + offset)
            successors.append(matrix.indices)
            offset += matrix.nnz
        z = model.observation_probabilities
        observation_cdf = np.cumsum(z / z.sum(axis=-1)[:, :, None], axis=-1, dtype=np.float64)
        observation_cdf /= observation_cdf[:, :, -1, None]
        cache["sampling"] = (
            np.concatenate(cumulative),
            np.asarray(pointers),
            np.concatenate(successors),
            np.ascontiguousarray(observation_cdf),
            np.searchsorted(np.unique(model.rewards), model.rewards),
        )
    return cache["sampling"]


@njit(cache=True, nogil=True)
def _rollout(
    rng,
    actions,
    writes,
    first_action,
    first_memory,
    epsilon,
    A,
    initial_cdf,
    transition_cdf,
    pointers,
    successors,
    observation_cdf,
    reward_indices,
    active,
    continuing,
    out_a,
    out_o,
    out_r,
    lengths,
):
    """Fixed per-trajectory RNG call order with searchsorted CDF draws."""
    N, H = out_a.shape
    for i in range(N):
        s = np.searchsorted(initial_cdf, rng.random(), side="right")
        m, o = 0, 0
        if not active[s]:
            continue
        length = 0
        for t in range(H):
            if t == 0:
                a, n = first_action, first_memory
            else:
                a, n = actions[t - 1, m, o], writes[t - 1, m, o]
            if rng.random() < epsilon:
                a = rng.integers(0, A)
            out_a[i, t] = a
            out_r[i, t] = reward_indices[s, a]
            start, end = pointers[a, s], pointers[a, s + 1]
            edge = start + np.searchsorted(
                transition_cdf[start:end], rng.random(), side="right"
            )
            s = successors[edge]
            o = np.searchsorted(observation_cdf[a, s], rng.random(), side="right")
            out_o[i, t] = o
            m = n
            length = t + 1
            if not continuing[a] or not active[s]:
                break
        lengths[i] = length


def sample_trajectories(
    problem: Problem,
    controller: Controller,
    n_trajectories: int,
    rng: np.random.Generator,
    *,
    exploration: float = 0.4,
) -> TrajectoryBatch:
    """Record action, next observation and reward index, with zero invalid padding.

    Exploration replaces only the action; retain the selected memory write.
    Reward indices follow np.unique(model.rewards). Terminal steps retain their
    reward and next observation. Initially terminal states have length zero.
    """
    _check_controller(problem, controller)
    if type(n_trajectories) is not int or n_trajectories < 1:
        raise ValueError("n_trajectories must be positive")
    if not np.isfinite(exploration) or not 0 <= exploration <= 1:
        raise ValueError("exploration must be between zero and one")
    model = problem.model
    initial_cdf = np.cumsum(model.initial_belief, dtype=np.float64)
    initial_cdf /= initial_cdf[-1]
    transition_cdf, pointers, successors, observation_cdf, reward_indices = (
        _prepare_sampling(model)
    )
    active, continuing = _masks(model)
    shape = (n_trajectories, problem.horizon)
    actions, observations, rewards = (np.zeros(shape, dtype=np.int64) for _ in range(3))
    lengths = np.zeros(n_trajectories, dtype=np.int64)
    with rng.bit_generator.lock:
        _rollout(
            rng,
            np.ascontiguousarray(controller.actions, dtype=np.int64),
            np.ascontiguousarray(controller.next_memory, dtype=np.int64),
            int(controller.initial_action),
            int(controller.initial_next_memory),
            float(exploration),
            model.num_actions,
            initial_cdf,
            transition_cdf,
            pointers,
            successors,
            observation_cdf,
            reward_indices,
            active,
            continuing,
            actions,
            observations,
            rewards,
            lengths,
        )
    return TrajectoryBatch(actions, observations, rewards, lengths)
