"""Finite-horizon MAPO search with independent streams and completed-candidate budgets."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import time

import numpy as np

from .em import fit_em
from .extraction import extract_controller
from .policy import policy_iteration, random_controller
from .sampling import sample_trajectories
from .io.results import provenance
from .types import (
    Controller, HalvingResult, Problem, SearchConfig, SearchFailure,
    SearchProgress, SearchResult, TrajectoryBatch,
)


def trajectory_count(problem: Problem, memory: int) -> int:
    """Paper rule: round max(2*S*M, 8*O*A) up to 100, clip to [400,4000]."""
    model = problem.model
    size = max(2 * model.num_states * memory,
               8 * model.num_observations * model.num_actions)
    return min(4000, max(400, 100 * ((size + 99) // 100)))


def _seed(problem, config, stage, stream, candidate=0, *, memory=None):
    # Hash-derived independent PCG64 streams, one per (stage, stream, candidate).
    key = [problem.name, config.memory if memory is None else memory,
           config.seed, stage, stream, candidate]
    payload = json.dumps(key, sort_keys=True, separators=(",", ":")).encode()
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def _check_config(config):
    for name, minimum in (("memory", 1), ("fits_per_round", 1),
                          ("reactive_starts", 1), ("max_sweeps", 1)):
        value = getattr(config, name)
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if config.initialization not in ("predictive", "random"):
        raise ValueError("initialization must be predictive or random")
    if config.cpu_seconds is not None and (
        not np.isfinite(config.cpu_seconds) or config.cpu_seconds < 0
    ):
        raise ValueError("cpu_seconds must be finite and nonnegative, or None")
    if config.rounds is not None and (
        not isinstance(config.rounds, (int, np.integer)) or config.rounds < 0
    ):
        raise ValueError("rounds must be nonnegative, or None")
    if config.cpu_seconds is None and config.rounds is None:
        raise ValueError("supply a CPU budget or a fixed round limit")
    if config.trajectories is not None and (
        not isinstance(config.trajectories, (int, np.integer)) or config.trajectories < 1
    ):
        raise ValueError("trajectories must be positive, or None")
    if not np.isfinite(config.exploration) or not 0 <= config.exploration <= 1:
        raise ValueError("exploration must be between zero and one")


class _Search:
    """In-memory state retained between the two halving allocations."""

    def __init__(self, problem, config):
        _check_config(config)
        self.problem = problem
        self.config = deepcopy(config)
        if self.config.trajectories is None:
            self.config.trajectories = trajectory_count(problem, config.memory)
        self.provenance = provenance(problem, config.seed)
        self.cpu = 0.0
        self.bootstrap = self.round = self.fit = 0
        self.progress = []
        self.failures = []
        self.incumbents = []  # Only record improvements, not every controller.
        self.best = self.best_value = self.best_index = None
        self.warm = self.warm_value = self.warm_index = None
        self.batch = None
        self.sampler_index = self.sampling_seed = None

    def _refine(self, initial):
        result = policy_iteration(self.problem, initial,
                                  max_sweeps=self.config.max_sweeps)
        if result.cycled:
            raise RuntimeError("policy iteration repeated a controller without convergence")
        if not result.converged:
            raise RuntimeError("policy iteration reached max_sweeps without convergence")
        return result.controller, result.value, result.sweeps

    def _record(self, controller, value, row):
        index = len(self.progress)
        self.progress.append(row)
        if self.best_value is None or value > self.best_value:
            self.best, self.best_value, self.best_index = controller, value, index
            self.incumbents.append((row.cpu_seconds, controller, value))
        return index

    def _elapsed(self):
        return self.cpu + time.process_time() - self.started

    def _complete(self, controller, origin, stage, fit, seed, fitted=None):
        controller, value, sweeps = self._refine(controller)
        if origin == "reactive" and self.config.memory > 1:
            # A reactive incumbent uses node zero at the requested M.
            controller = Controller(
                np.repeat(controller.actions, self.config.memory, axis=1),
                np.zeros((self.problem.horizon - 1, self.config.memory,
                          self.problem.model.num_observations), dtype=np.int64),
                controller.initial_action, 0,
            )
        row = SearchProgress(
            cpu_seconds=self._elapsed(), value=value, origin=origin,
            round=stage, fit=fit, seed=seed, sweeps=sweeps,
            sampler=self.sampler_index if fitted is not None else None,
            sampling_seed=self.sampling_seed if fitted is not None else None,
            warm_source=self.warm_index if fitted is not None and stage > 0 and fit == 0 else None,
            em_iterations=fitted.iterations if fitted is not None else 0,
            em_stop_reason=fitted.stop_reason if fitted is not None else None,
        )
        index = self._record(controller, value, row)
        if fitted is not None and (self.warm_value is None or value > self.warm_value):
            self.warm, self.warm_value, self.warm_index = fitted.model, value, index

    def _bootstrap_candidate(self, seed):
        controller = random_controller(self.problem, 1, np.random.default_rng(seed))
        self._complete(controller, "reactive", -1, self.bootstrap, seed)

    def _random_candidate(self, seed):
        controller = random_controller(self.problem, self.config.memory,
                                       np.random.default_rng(seed))
        self._complete(controller, "random", self.round, 0, seed)

    def _sample_batch(self):
        if self.best is None:
            raise RuntimeError("no converged reactive sampling controller")
        self.sampler_index = self.best_index
        self.sampling_seed = _seed(self.problem, self.config, self.round, "sampling")
        return sample_trajectories(
            self.problem, self.best, self.config.trajectories,
            np.random.default_rng(self.sampling_seed), exploration=self.config.exploration,
        )

    def _predictive_candidate(self, seed, batch: TrajectoryBatch):
        initial = self.warm if self.round > 0 and self.fit == 0 else None
        fitted = fit_em(self.problem, batch, self.config.memory, seed=seed,
                        config=self.config.em, initial=initial)
        if fitted.stop_reason == "numerical_failure":
            raise RuntimeError("EM numerical_failure")
        controller = extract_controller(self.problem, fitted.model)
        self._complete(controller, "predictive", self.round, self.fit, seed, fitted)

    def advance(self, cutoff):
        """Continue until cutoff; retain unfinished rounds for halving continuation."""
        config, problem = self.config, self.problem
        self.started = time.process_time()
        try:
            while self._elapsed() < cutoff:
                reactive = (config.initialization == "predictive"
                            and self.bootstrap < config.reactive_starts)
                if not reactive and config.rounds is not None and self.round >= config.rounds:
                    break
                origin = "reactive" if reactive else config.initialization
                stage, fit = (-1, self.bootstrap) if reactive else (self.round, self.fit)
                stream = "fit" if origin == "predictive" else origin
                seed = _seed(problem, config, stage, stream,
                             fit if origin != "random" else 0,
                             memory=1 if reactive else None)
                try:
                    if reactive:
                        self._bootstrap_candidate(seed)
                    elif origin == "random":
                        self._random_candidate(seed)
                    else:
                        if self.batch is None:
                            self.batch = self._sample_batch()
                            if self._elapsed() >= cutoff:
                                break  # Sampling is charged; don't start a late fit.
                        self._predictive_candidate(seed, self.batch)
                except (RuntimeError, FloatingPointError) as error:
                    self.failures.append(SearchFailure(self._elapsed(), origin,
                                                       stage, fit, seed, str(error)))
                    if origin == "predictive" and self.batch is None:
                        self.fit = config.fits_per_round - 1
                if reactive:
                    self.bootstrap += 1
                elif origin == "random":
                    self.round += 1
                else:
                    self.fit += 1
                    if self.fit == config.fits_per_round:
                        self.round += 1
                        self.fit = 0
                        self.batch = None
        finally:
            self.cpu += time.process_time() - self.started

    def result(self, cutoff):
        controller = value = None
        for completed, candidate, candidate_value in self.incumbents:
            if completed <= cutoff:
                controller, value = candidate, candidate_value
        return SearchResult(
            controller, value, self.cpu,
            self.progress.copy(),
            replace(self.config, cpu_seconds=None if np.isinf(cutoff) else cutoff),
            self.provenance, self.failures.copy(),
        )


def search(problem: Problem, config: SearchConfig | None = None) -> SearchResult:
    """Run predictive or random MAPO search. CPU includes all solver preparation.

    Use cpu_seconds=None and rounds=N for deterministic fixed work. Reactive
    bootstrap runs are included in the budget. Late candidates are traced but
    cannot become the returned incumbent. No interrupted-run resume is provided.
    """
    state = _Search(problem, SearchConfig() if config is None else config)
    cutoff = np.inf if state.config.cpu_seconds is None else state.config.cpu_seconds
    state.advance(cutoff)
    return state.result(cutoff)


def successive_halving(problem: Problem, config: SearchConfig | None = None,
                       memory_sizes=(2, 4, 8, 16)) -> HalvingResult:
    """Run each M to B/2, then continue the best half to B (default total 3B).

    State and already spent CPU are retained in memory; survivors aren't restarted.
    Ranking uses only candidates completed by the allocation cutoff. Ties prefer
    smaller M. The returned winner may come from an eliminated arm's eligible result.
    """
    config = SearchConfig() if config is None else config
    _check_config(config)
    if config.cpu_seconds is None or config.rounds is not None:
        raise ValueError("successive halving requires a CPU budget and no round cap")
    sizes = sorted(memory_sizes)
    if len(sizes) < 2 or len(set(sizes)) != len(sizes) or any(m < 1 for m in sizes):
        raise ValueError("supply at least two unique positive memory sizes")
    states = [_Search(problem, replace(config, memory=m)) for m in sizes]
    half = config.cpu_seconds / 2
    for state in states:
        state.advance(half)
    first = [state.result(half) for state in states]
    def rank(arm):
        value = arm.value
        return value if value is not None else -np.inf, -arm.config.memory

    ranking = sorted(range(len(sizes)), key=lambda i: rank(first[i]), reverse=True)
    survivors = ranking[: (len(sizes) + 1) // 2]
    for i in survivors:
        states[i].advance(config.cpu_seconds)
    arms = [state.result(config.cpu_seconds if i in survivors else half)
            for i, state in enumerate(states)]
    eligible = [arm for arm in arms if arm.value is not None]
    best = max(eligible, key=rank) if eligible else None
    return HalvingResult(arms, [arm.value for arm in first],
                         [sizes[i] for i in survivors],
                         best.config.memory if best is not None else None,
                         sum(state.cpu for state in states))
