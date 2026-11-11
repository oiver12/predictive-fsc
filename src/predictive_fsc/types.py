"""Data containers for the finite-horizon predictive-FSC pipeline.

Constructors store values directly, without validation, copying or conversion.
Indices are zero-based; no observation exists at t=0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.int64]
EMStopReason = Literal["converged", "iteration_limit", "numerical_failure"]


@dataclass(eq=False)
class POMDP:
    """T[a][s,s'], Z[a,s',o'], r[s,a], b0[s]; one CSR matrix per action.

    Names are resolved by the loader; counts come from array dimensions.
    Terminal actions and states are explicit indices. Absorbing states are not
    automatically terminal. A terminal step's reward/observation remain valid.
    """

    transitions: list[csr_matrix]
    observation_probabilities: FloatArray
    rewards: FloatArray
    initial_belief: FloatArray
    terminal_actions: list[int] = field(default_factory=list)
    terminal_states: list[int] = field(default_factory=list)
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    @property
    def num_states(self) -> int:
        return self.rewards.shape[0]

    @property
    def num_actions(self) -> int:
        return self.rewards.shape[1]

    @property
    def num_observations(self) -> int:
        return self.observation_probabilities.shape[2]


@dataclass
class Problem:
    name: str
    model: POMDP
    horizon: int


@dataclass(eq=False)
class Controller:
    """Tables [t-1,m,o] for t=1..H-1. At t=0 use the scalar entries.

    Select (action,next_memory) together before observing the next observation.
    Memory starts at zero. Terminal/horizon-ending writes have no control effect.
    """

    actions: IndexArray
    next_memory: IndexArray
    initial_action: int
    initial_next_memory: int

    @property
    def horizon(self) -> int:
        return self.actions.shape[0] + 1

    @property
    def num_memory(self) -> int:
        return self.actions.shape[1]


@dataclass(eq=False)
class TrajectoryBatch:
    """Entry [i,t] is (a_t,o_{t+1},reward_index_{t+1}); no initial o0.

    Only t < lengths[i] is valid. Remaining entries are padding, filled with zero
    by the sampler and ignored by EM.
    Length zero is allowed (e.g. an initially terminal state); EM ignores it.
    """

    actions: IndexArray
    observations: IndexArray
    reward_indices: IndexArray
    lengths: IndexArray


@dataclass(eq=False)
class PredictiveModel:
    """Shared-stage IOHMM parameters; each final axis is a distribution.

    B[m,o,a,m'], EO[m,o,a,o'], ER[m,o,a,r_index]. Initial arrays
    B0[a,m'], EO0[a,o'], ER0[a,r_index] describe the separate first step.
    """

    memory_transitions: FloatArray
    observation_probabilities: FloatArray
    reward_probabilities: FloatArray
    initial_memory_probabilities: FloatArray
    initial_observation_probabilities: FloatArray
    initial_reward_probabilities: FloatArray
    reward_values: FloatArray


@dataclass
class EMConfig:
    max_iterations: int = 100
    tolerance: float = 1e-5
    pseudocount: float = 0.1


@dataclass
class EMResult:
    model: PredictiveModel
    iterations: int
    log_likelihoods: list[float]
    objectives: list[float]
    stop_reason: EMStopReason


@dataclass
class SearchConfig:
    memory: int = 4
    cpu_seconds: float | None = 75.0
    seed: int = 0
    initialization: Literal["predictive", "random"] = "predictive"
    trajectories: int | None = None  # None selects the paper's size rule.
    fits_per_round: int = 4
    reactive_starts: int = 300
    exploration: float = 0.4
    em: EMConfig = field(default_factory=EMConfig)
    rounds: int | None = None  # Optional fixed-work cap; random uses one start/round.
    max_sweeps: int = 100_000


@dataclass
class SearchProgress:
    """A completed candidate; compare its CPU time with the result cutoff for eligibility."""

    cpu_seconds: float
    value: float
    origin: Literal["reactive", "predictive", "random"]
    round: int = -1
    fit: int = 0
    seed: int = 0
    sweeps: int = 0
    sampler: int | None = None  # Index in progress of the sampling controller.
    sampling_seed: int | None = None
    warm_source: int | None = None  # Index of the best EM-derived controller.
    em_iterations: int = 0
    em_stop_reason: EMStopReason | None = None


@dataclass
class SearchFailure:
    cpu_seconds: float
    origin: str
    round: int
    fit: int
    seed: int
    message: str


@dataclass
class RunProvenance:
    problem_sha256: str
    source_revision: str
    seed: int
    build_identity: dict[str, str] = field(default_factory=dict)


@dataclass
class SearchResult:
    """An incumbent completed within budget. No eligible incumbent => None/value None."""

    controller: Controller | None
    value: float | None
    cpu_seconds: float
    progress: list[SearchProgress]
    config: SearchConfig
    provenance: RunProvenance
    failures: list[SearchFailure] = field(default_factory=list)

    @property
    def eligible(self) -> list[SearchProgress]:
        """Candidates completed within the CPU budget."""
        cutoff = self.config.cpu_seconds
        return [row for row in self.progress if cutoff is None or row.cpu_seconds <= cutoff]


@dataclass
class HalvingResult:
    arms: list[SearchResult]
    half_values: list[float | None]
    survivors: list[int]
    selected_memory: int | None
    cpu_seconds: float

    @property
    def best(self) -> SearchResult | None:
        return next((arm for arm in self.arms
                     if arm.config.memory == self.selected_memory), None)


@dataclass
class LoadedProblem:
    """File-layer labels accompanying the numerical problem."""

    problem: Problem
    states: list[str]
    actions: list[str]
    observations: list[str]


@dataclass
class PolicyIterationResult:
    controller: Controller
    value: float
    sweeps: int
    converged: bool
    sweep_values: list[float]
    cycled: bool = False
