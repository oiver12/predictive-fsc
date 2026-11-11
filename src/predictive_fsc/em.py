"""One complete C++ EM fit for the paper's finite-horizon predictive model."""

import numpy as np

from ._em import fit as _fit
from .types import EMConfig, EMResult, PredictiveModel, Problem, TrajectoryBatch


def fit_em(
    problem: Problem,
    trajectories: TrajectoryBatch,
    memory: int,
    *,
    seed: int = 0,
    config: EMConfig | None = None,
    initial: PredictiveModel | None = None,
) -> EMResult:
    """Fit the shared-stage IOHMM, using random labels or the supplied warm start.

    C++ owns initialization, scaled forward/backward, M-steps and stopping.
    The objective is likelihood plus the symmetric Dirichlet pseudocount prior.
    Histories include initialization and accepted updates. Inputs are unchanged.
    """
    config = EMConfig() if config is None else config
    # Index ranges, shapes and warm-start probabilities are validated in C++.
    reward_values = np.unique(problem.model.rewards).astype(np.float64, copy=False)
    parameters = None if initial is None else (
        initial.memory_transitions,
        initial.observation_probabilities,
        initial.reward_probabilities,
        initial.initial_memory_probabilities,
        initial.initial_observation_probabilities,
        initial.initial_reward_probabilities,
    )
    fitted, iterations, likelihoods, objectives, reason = _fit(
        trajectories.actions,
        trajectories.observations,
        trajectories.reward_indices,
        trajectories.lengths,
        memory,
        problem.model.num_observations,
        problem.model.num_actions,
        len(reward_values),
        config.max_iterations,
        config.tolerance,
        config.pseudocount,
        seed,
        parameters,
    )
    model = PredictiveModel(*fitted, reward_values.copy())
    return EMResult(model, iterations, likelihoods, objectives, reason)
