from __future__ import annotations

from typing import Generator, Optional

import torch
from torch import Tensor

from ..operators import LinearOperator, LowRankRootLinearOperator, ZeroLinearOperator, to_linear_operator
from .linear_solver import LinearSolver, LinearSolverState, LinearSystem


class PLS(LinearSolver):
    """Probabilistic linear solver.

    Iteratively solve linear systems of the form

    .. math:: Ax_* = b

    where :math:`A` is a (symmetric positive-definite) linear operator. A probabilistic
    linear solver chooses actions :math:`s_i` in each iteration to observe the residual
    by computing :math:`\\alpha_i = s_i^\\top (b - Ax_i)`.

    :param policy: Policy selecting actions :math:`s_i` to probe the residual with.
    :param abstol: Absolute residual tolerance.
    :param reltol: Relative residual tolerance.
    :max_iter: Maximum number of iterations. Defaults to `10 * rhs.shape[0]`.
    """

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
        linear_op: LinearOperator,
        rhs: Tensor,
        /,
        x: Optional[Tensor] = None,
    ) -> Generator[LinearSolverState, None, None]:
        r"""Generator implementing the linear solver iteration.

        This function allows stepping through the solver iteration one step at a time and thus exposes internal quantities in the solver state cache.

        :param linear_op: Linear operator :math:`A`.
        :param rhs: Right-hand-side :math:`b`.
        :param x: Initial guess :math:`x \approx x_*`.
        """
        # Setup
        linear_op = to_linear_operator(linear_op)
        if self.max_iter is None:
            max_iter = 10 * rhs.shape[0]
        else:
            max_iter = self.max_iter

        if x is None:
            x = torch.zeros_like(rhs, requires_grad=True)
            inverse_op = ZeroLinearOperator(*linear_op.shape, dtype=linear_op.dtype, device=linear_op.device)
            residual = rhs
            logdet = torch.zeros((), requires_grad=True)
        else:
            # Construct a better initial guess with a consistent inverse approximation such that x = inverse_op @ rhs
            action = x
            linear_op_action = linear_op @ action
            action_linear_op_action = linear_op_action.T @ action

            # Potentially improved initial guess x derived from initial guess
            step_size = action.T @ rhs / action_linear_op_action
            x = step_size * action

            # Initial residual
            linear_op_x = step_size * linear_op_action
            residual = rhs - linear_op_x

            # Consistent inverse approximation for new initial guess
            inverse_op = LowRankRootLinearOperator((action / torch.sqrt(action_linear_op_action)).reshape(-1, 1))

            # Log determinant
            logdet = torch.log(action_linear_op_action)

        # Initialize solver state
        solver_state = LinearSolverState(
            problem=LinearSystem(A=linear_op, b=rhs),
            solution=x,
            forward_op=None,
            inverse_op=inverse_op,
            residual=residual,
            residual_norm=torch.linalg.vector_norm(residual, ord=2),
            logdet=logdet,
            iteration=0,
            cache={
                "search_dir_sq_Anorms": [],
                "rhs_norm": torch.linalg.vector_norm(rhs, ord=2),
                "action": None,
                "observation": None,
                "search_dir": None,
                "step_size": None,
            },
        )

        yield solver_state

        while True:

            # Check convergence
            if (
                solver_state.residual_norm < max(self.abstol, self.reltol * solver_state.cache["rhs_norm"])
                or solver_state.iteration >= max_iter
            ):
                break

            # Select action
            action = self.policy(solver_state)
            linear_op_action = linear_op @ action

            # Observation
            observ = action.T @ solver_state.residual

            # Search direction
            if isinstance(solver_state.inverse_op, ZeroLinearOperator):
                search_dir = action
            else:
                search_dir = action - solver_state.inverse_op @ linear_op_action

            # Normalization constant
            search_dir_sqnorm = linear_op_action.T @ search_dir
            solver_state.cache["search_dir_sq_Anorms"].append(search_dir_sqnorm)

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
            solver_state.residual = rhs - linear_op @ solver_state.solution
            solver_state.residual_norm = torch.linalg.vector_norm(solver_state.residual, ord=2)

            # Update log-determinant
            solver_state.logdet = solver_state.logdet + torch.log(search_dir_sqnorm)

            # Update iteration
            solver_state.iteration += 1

            # Update solver state cache
            solver_state.cache["action"] = action
            solver_state.cache["observation"] = observ
            solver_state.cache["search_dir"] = search_dir
            solver_state.cache["step_size"] = step_size

            yield solver_state

    def solve(
        self,
        linear_op: LinearOperator,
        rhs: Tensor,
        /,
        x: Optional[Tensor] = None,
    ) -> LinearSolverState:
        r"""Solve linear system :math:`Ax_*=b`.

        :param linear_op: Linear operator :math:`A`.
        :param rhs: Right-hand-side :math:`b`.
        :param x: Initial guess :math:`x \approx x_*`.
        """

        solver_state = None

        for solver_state in self.solve_iterator(linear_op, rhs, x=x):
            pass

        return solver_state