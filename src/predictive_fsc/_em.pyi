"""Type stubs for the compiled EM extension."""

from typing import TypeAlias
import numpy as np
from .types import EMStopReason, FloatArray

compiler: str

_Parameters: TypeAlias = tuple[
    FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray
]

def fit(
    actions: np.ndarray,
    observations: np.ndarray,
    rewards: np.ndarray,
    lengths: np.ndarray,
    memory: int,
    num_observations: int,
    num_actions: int,
    num_rewards: int,
    max_iterations: int,
    tolerance: float,
    pseudocount: float,
    seed: int,
    initial: tuple[np.ndarray, ...] | None,
) -> tuple[_Parameters, int, list[float], list[float], EMStopReason]: ...
