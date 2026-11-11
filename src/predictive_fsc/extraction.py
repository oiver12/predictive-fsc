"""Deterministic memory routing followed by predictive backward induction."""

import numpy as np
from numba import njit

from .types import Controller, PredictiveModel, Problem


@njit(cache=True, nogil=True)
def _backward(rewards, observations, routing, initial_rewards,
              initial_observations, initial_routing, terminal, horizon):
    M, O, A = rewards.shape
    actions = np.empty((horizon - 1, M, O), dtype=np.int64)
    writes = np.empty_like(actions)
    following = np.zeros((M, O))
    for t in range(horizon - 2, -1, -1):
        values = np.empty((M, O))
        for m in range(M):
            for o in range(O):
                best, chosen = -np.inf, 0
                for a in range(A):
                    n = routing[m, o, a]
                    q = rewards[m, o, a]
                    if not terminal[a]:
                        for on in range(O):
                            q += observations[m, o, a, on] * following[n, on]
                    if q > best:
                        best, chosen = q, a
                actions[t, m, o] = chosen
                writes[t, m, o] = routing[m, o, chosen]
                values[m, o] = best
        following = values
    best, first = -np.inf, 0
    for a in range(A):
        q = initial_rewards[a]
        if not terminal[a]:
            for o in range(O):
                q += initial_observations[a, o] * following[initial_routing[a], o]
        if q > best:
            best, first = q, a
    return actions, writes, first, initial_routing[first]


def extract_controller(problem: Problem, model: PredictiveModel) -> Controller:
    """Fix each most-likely memory write, then optimize the finite-horizon actions.

    Exact ties choose the first memory/action index.
    Terminal actions have no continuation. Terminal-state behavior is learned
    through the emissions; known-state terminal masks apply during refinement.
    """
    A = problem.model.num_actions
    routing = np.argmax(model.memory_transitions, axis=-1)
    initial_routing = np.argmax(model.initial_memory_probabilities, axis=-1)
    terminal = np.zeros(A, dtype=np.bool_)
    terminal[problem.model.terminal_actions] = True
    result = _backward(model.reward_probabilities @ model.reward_values,
                       model.observation_probabilities, routing,
                       model.initial_reward_probabilities @ model.reward_values,
                       model.initial_observation_probabilities, initial_routing,
                       terminal, problem.horizon)
    return Controller(*result)
