"""Transition preparation and compact Bellman/occupancy operations."""

import numpy as np


def canonical(matrix):
    """Copy of a CSR matrix with summed duplicates, sorted indices and no stored zeros."""
    matrix = matrix.tocsr(copy=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    matrix.eliminate_zeros()
    return matrix


def prepare_transitions(model):
    """Automatic rule: CSR for large sparse models, dense otherwise."""
    cache = model._cache  # Models are not modified after loading.
    if "transitions" not in cache:
        S, A = model.num_states, model.num_actions
        sparse = S >= 128 and sum(t.nnz for t in model.transitions) <= 0.1 * A * S * S
        if sparse:
            transitions = tuple(model.transitions)
            transposes = tuple(t.T.tocsr() for t in transitions)
        else:
            transitions = np.stack([t.toarray() for t in model.transitions])
            transposes = transitions.transpose(0, 2, 1)
        cache["transitions"] = transitions, transposes, sparse
    return cache["transitions"]


class Dynamics:
    """Per-call view of cached transitions; compact values carry (S,M,A_prev)."""

    def __init__(self, model, active, continuing):
        self.model = model
        self.transitions, self.transposes, self.sparse = prepare_transitions(model)
        self.active, self.continuing = active, continuing

    def q_values(self, following):
        model = self.model
        if self.sparse:
            future = np.empty((model.num_states, following.shape[1], model.num_actions))
            for a, t in enumerate(self.transitions):
                future[:, :, a] = t @ (
                    following if following.ndim == 2 else following[:, :, a]
                )
        elif following.ndim == 2:
            future = np.moveaxis(self.transitions @ following, 0, 2)
        else:
            values = np.ascontiguousarray(np.moveaxis(following, 2, 0))
            future = np.moveaxis(self.transitions @ values, 0, 2)
        # The product buffer is owned here, so add reward/mask in place.
        if model.terminal_actions:
            future *= self.continuing[None, None, :]
        future += model.rewards[:, None, :]
        q = future.transpose(0, 2, 1)
        if model.terminal_states:
            q[~self.active] = 0
        return q  # (S,A,M_next)

    def initial_mass(self, action, memory, M):
        mass = np.zeros((self.model.num_states, M, self.model.num_actions))
        if self.continuing[action]:
            mass[:, memory, action] = self.transposes[action] @ (
                self.model.initial_belief * self.active
                if self.model.terminal_states
                else self.model.initial_belief
            )
            if self.model.terminal_states:
                mass[~self.active] = 0
        return mass

    def forward(self, joint, actions, writes):
        """Group observation/memory columns by action, multiply, then scatter."""
        S, M, O = joint.shape
        A = self.model.num_actions
        beta = joint.transpose(0, 2, 1).reshape(S, O * M)
        selected = actions.T.ravel()
        successors = np.zeros((S, O * M))
        for a in range(A):
            columns = np.flatnonzero(selected == a)
            if columns.size:
                successors[:, columns] = self.transposes[a] @ beta[:, columns]
        destinations = (writes.T * A + actions.T).ravel()
        scatter = np.zeros((O * M, M * A))
        scatter[np.arange(O * M), destinations] = self.continuing[selected]
        mass = (successors @ scatter).reshape(S, M, A)
        if self.model.terminal_states:
            mass[~self.active] = 0
        return mass
