from __future__ import annotations

from typing import Optional

import torch

from ...operators import LinearOperator
from ...utils.lanczos import lanczos_tridiag, lanczos_tridiag_to_diag
from .linear_solver_policy import LinearSolverPolicy


class AdverserialPolicy(LinearSolverPolicy):
    """Policy choosing actions that exclusively reduce computational uncertainty but not update the mean."""

    def __init__(
        self,
        base_policy: LinearSolverPolicy,
        num_non_zero: Optional[int] = None,
    ) -> None:
        self.base_policy = base_policy
        self.num_nonzero = num_non_zero
        super().__init__()

    def __call__(self, solver_state: "LinearSolverState") -> torch.Tensor:
        with torch.no_grad():
            n = solver_state.problem.A.shape[0]
            y = solver_state.problem.b

            if solver_state.iteration < n - 1:
                # Arbitrary linearly independent action sequence
                action = self.base_policy(solver_state=solver_state)

                # Enforce orthogonality to observations
                action = action - y * (y.T @ action) / (y.T @ y)

                # Sparsify
                if self.num_nonzero is not None:
                    topk_vals, topk_idcs = torch.topk(
                        torch.abs(action), k=self.num_nonzero, largest=True
                    )
                    action = torch.zeros(
                        solver_state.problem.A.shape[0],
                        dtype=solver_state.problem.A.dtype,
                        device=solver_state.problem.A.device,
                    )
                    action[topk_idcs] = topk_vals

                return action

            return y
