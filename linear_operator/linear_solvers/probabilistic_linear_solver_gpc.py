from __future__ import annotations

from typing import Generator, Optional
from warnings import warn

import torch
from torch import Tensor

from .. import settings
from ..operators import (
    LinearOperator,
    LowRankRootLinearOperator,
    ZeroLinearOperator,
    to_linear_operator,
)
from .linear_solver import LinearSolver, LinearSolverState, LinearSystem


class PLS_GPC(LinearSolver):
    """Probabilistic linear solver specifically designed for GP classification."""

    def __init__(
        self,
        policy: "LinearSolverPolicy",
        abstol: float = 1e-5,
        reltol: float = 1e-5,
        max_iter: int = None,
    ):
        self.policy = policy
        self.abstol = abstol
        self.reltol = reltol
        self.max_iter = max_iter

    def solve_iterator(
        self,
        K_op: LinearOperator,
        Winv_op: LinearOperator,
        rhs: Tensor,
        *,  # enforce keyword arguments in the following
        x: Optional[Tensor] = None,
        actions: Optional[Tensor] = None,  # N x i tensor
        K_op_actions: Optional[Tensor] = None,  # N x i tensor
    ) -> Generator[LinearSolverState, None, None]:
        r"""Generator implementing the linear solver iteration.

        This function allows stepping through the solver iteration one step at a time
        and thus exposes internal quantities in the solver state cache.
        """

        # Convert to linear operator
        K_op = to_linear_operator(K_op)
        Winv_op = to_linear_operator(Winv_op)

        # Initialize `max_iter` if not given
        if self.max_iter is None:
            max_iter = 10 * rhs.shape[0]
        else:
            max_iter = self.max_iter

        # Ensure rhs is vector
        rhs = rhs.reshape(-1)
        if x is not None:
            x = x.reshape(-1)

        # There are three cases: (1) `actions` and `K_op_actions` are given. Then, we
        # construct `inverse_op` and compute a consistent initial solution (i.e. `x` can
        # not be used and must be `None`). If this is not the case, we dont' use
        # preconditioning. In this case, we have to distinguish two sub-cases: (2) `x`
        # is not given and (3) `x` is given. These two cases were already implemented in
        # `PLS` and are thus copied.
        if (actions is not None) and (K_op_actions is not None):  # case (1)
            assert x is None

            print("[PLS_GPC] Case 1: Preconditioning")

            # Check dimensions of tensors
            assert K_op_actions.shape == actions.shape  # both: N x i
            assert K_op_actions.shape[0] == K_op.shape[0]

            # Compute `M`
            M = actions.T @ K_op_actions + actions.T @ Winv_op @ actions

            # Compute its inverse via SVD (`M` is spd)
            Lambda_diag, U = torch.linalg.eigh(M)
            Root = actions @ U @ torch.diag(torch.sqrt(1 / Lambda_diag))
            inverse_op = LowRankRootLinearOperator(Root)

            # Compute consistent solution and residual
            solution = inverse_op @ rhs
            residual = rhs - (K_op + Winv_op) @ solution
        else:
            assert (actions is None) and (K_op_actions is None)

            if x is None:  # case (2)
                print("[PLS_GPC] Case 2: Setting `x` to zero")

                # "Trivial" initialization
                inverse_op = ZeroLinearOperator(
                    *K_op.shape, dtype=K_op.dtype, device=K_op.device
                )
                solution = torch.zeros_like(rhs)
                residual = rhs

            else:  # case (3)
                print("[PLS_GPC] Case 3: Using given `x`")

                # Construct a better initial guess with a consistent inverse
                # approximation such that x = inverse_op @ rhs
                action = x
                linear_op_action = (K_op + Winv_op) @ action
                action_linear_op_action = torch.inner(linear_op_action, action)

                # Potentially improved initial guess x derived from initial guess
                step_size = torch.inner(action, rhs) / action_linear_op_action
                solution = step_size * action

                # Initial residual
                linear_op_x = step_size * linear_op_action
                residual = rhs - linear_op_x

                # Consistent inverse approximation for new initial guess
                Root = (action / torch.sqrt(action_linear_op_action))
                Root = Root.reshape(-1, 1)
                inverse_op = LowRankRootLinearOperator(Root)

        # Initialize solver state
        solver_state = LinearSolverState(
            problem=LinearSystem(A=K_op + Winv_op, b=rhs),
            solution=solution,
            forward_op=None,
            inverse_op=inverse_op,
            residual=residual,
            residual_norm=torch.linalg.vector_norm(residual, ord=2),
            logdet=None,
            iteration=0,
            cache={
                "search_dir_sq_Anorms": [],
                "rhs_norm": torch.linalg.vector_norm(rhs, ord=2),
                "action": None,
                "observation": None,
                "search_dir": None,
                "step_size": None,
                "actions": None,
                "K_op_actions": None,
            },
        )

        yield solver_state

        while True:
            # Check convergence
            if (
                solver_state.residual_norm
                < max(self.abstol, self.reltol * solver_state.cache["rhs_norm"])
                or solver_state.iteration >= max_iter
            ):
                break

            # Select action
            action = self.policy(solver_state)

            # Evaluate matrix-vector product with `K_op` seperately
            K_op_action = K_op @ action

            # Evaluate `(K_op + Winv_op) @ action`
            linear_op_action = K_op_action + Winv_op @ action

            # Observation
            observ = torch.inner(action, solver_state.residual)

            # Search direction
            if isinstance(solver_state.inverse_op, ZeroLinearOperator):
                search_dir = action
            else:
                search_dir = action - solver_state.inverse_op @ linear_op_action

            # Normalization constant
            search_dir_sqnorm = torch.inner(linear_op_action, search_dir)
            solver_state.cache["search_dir_sq_Anorms"].append(search_dir_sqnorm)

            if search_dir_sqnorm <= 0:
                if settings.verbose_linalg.on():
                    settings.verbose_linalg.logger.debug(
                        f"PLS terminated after {solver_state.iteration} iteration(s)"
                        + " due to a negative normalization constant."
                    )
                warn_msg = f"""
                PLS terminated after {solver_state.iteration} iteration(s) due to a
                negative normalization constant. This can happen, e.g. when the current
                action has already been used in the preconditioner.
                """
                warn(warn_msg)
                break

            # Update solution estimate
            step_size = observ / search_dir_sqnorm
            solver_state.solution = solver_state.solution + step_size * search_dir

            # Update inverse approximation
            if isinstance(solver_state.inverse_op, ZeroLinearOperator):
                solver_state.inverse_op = LowRankRootLinearOperator(
                    (search_dir / torch.sqrt(search_dir_sqnorm)).reshape(-1, 1)
                )
            else:
                solver_state.inverse_op = LowRankRootLinearOperator(
                    torch.concat(
                        (
                            solver_state.inverse_op.root.to_dense(),
                            (search_dir / torch.sqrt(search_dir_sqnorm)).reshape(-1, 1),
                        ),
                        dim=1,
                    )
                )

            # Update residual
            solver_state.residual = rhs - (K_op + Winv_op) @ solver_state.solution
            solver_state.residual_norm = torch.linalg.vector_norm(
                solver_state.residual, ord=2
            )

            # Update iteration
            solver_state.iteration += 1

            # Update solver state cache
            solver_state.cache["action"] = action
            solver_state.cache["observation"] = observ
            solver_state.cache["search_dir"] = search_dir
            solver_state.cache["step_size"] = step_size

            # Append `action` to `actions`
            if solver_state.cache["actions"] is None:
                solver_state.cache["actions"] = action.reshape(-1, 1)
            else:
                solver_state.cache["actions"] = torch.hstack(
                    (solver_state.cache["actions"], action.reshape(-1, 1))
                )

            # Append `K_op_action` to `K_op_actions`
            if solver_state.cache["K_op_actions"] is None:
                solver_state.cache["K_op_actions"] = K_op_action.reshape(-1, 1)
            else:
                solver_state.cache["K_op_actions"] = torch.hstack(
                    (solver_state.cache["K_op_actions"], K_op_action.reshape(-1, 1))
                )

            yield solver_state

    def solve(self):
        raise NotImplementedError()
