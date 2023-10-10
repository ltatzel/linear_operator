"""Linear solver based on a Cholesky decomposition."""


from typing import Optional, Generator, Any, Union

import torch
from torch import Tensor, Size

from linear_operator.operators import LinearOperator

from ..operators import LinearOperator, ZeroLinearOperator
from .linear_solver import LinearSolver, LinearSolverState, LinearSystem


class _CholeskyFactorInverseLinearOperator(LinearOperator):
    def __init__(
        self,
        cholfac: Tensor,
        is_upper_cholfac: bool = False,
        linear_solver: Any or None = None,
        **kwargs,
    ):
        super().__init__(
            cholfac,
            is_upper_cholfac=is_upper_cholfac,
            linear_solver=linear_solver,
            **kwargs,
        )
        self.cholfac = cholfac
        self.is_upper_cholfac = is_upper_cholfac

    def _matmul(self, rhs: Tensor) -> Tensor:
        return torch.linalg.solve_triangular(
            self.cholfac.T, rhs, upper=not self.is_upper_cholfac, left=True
        )

    def _size(self) -> Size:
        return self.cholfac.size()

    def _transpose_nonbatch(self) -> LinearOperator:
        return _CholeskyFactorInverseLinearOperator(
            cholfac=self.cholfac.T, is_upper_cholfac=not self.is_upper_cholfac
        )


class CholeskySolveLinearOperator(LinearOperator):
    r"""Linear operator performing a linear solve :math:`v \mapsto A^{-1}v` via a Cholesky decomposition :math:`A=LL^{\top}`.

    Assumes :math:`A` is symmetric positive (semi-)definite.
    """

    def __init__(
        self,
        a: Union[Tensor, LinearOperator],
        **kwargs,
    ):
        if isinstance(a, LinearOperator):
            self.a = a.to_dense()
        elif isinstance(a, torch.Tensor):
            self.a = a

        # Cholesky decomposition
        self.cholfac = torch.linalg.cholesky(self.a, upper=False)
        self.root = _CholeskyFactorInverseLinearOperator(
            self.cholfac, is_upper_cholfac=False
        )

        super().__init__(
            a,
            **kwargs,
        )

    def _matmul(self, rhs: Tensor) -> Tensor:
        return torch.cholesky_solve(rhs, self.cholfac, upper=False)

    def to_dense(self):
        return torch.cholesky_inverse(self.cholfac, upper=False)

    def _size(self) -> Size:
        return self.a.size()

    def _transpose_nonbatch(self) -> LinearOperator:
        return self


class Cholesky_GPC(LinearSolver):
    r"""Linear solver performing a Cholesky decomposition.

    Solves a linear system

    .. math:: Ax_* = b

    where :math:`A` is a symmetric positive-definite matrix by performing a Cholesky
    decomposition of :math:`A` and then performing a forward and backward substitution.
    """

    def __init__(self) -> None:
        super().__init__()

    def solve(
        self, linear_op: LinearOperator, rhs: Tensor, /, **kwargs
    ) -> LinearSolverState:
        return super().solve(linear_op, rhs, **kwargs)

    def solve_iterator(
        self,
        K_op: LinearOperator,
        Winv_op: LinearOperator,
        rhs: Tensor,
        *,  # enforce keyword arguments in the following
        x: Optional[Tensor] = None,
        actions: Optional[Tensor] = None,  # N x i tensor
        K_op_actions: Optional[Tensor] = None,  # N x i tensor
        top_k: Optional[int] = None,
        kappa: Optional[float] = None,
    ) -> Generator[LinearSolverState, None, None]:
        r"""Generator implementing the linear solver iteration.

        This function allows stepping through the solver iteration one step at a time
        and thus exposes internal quantities in the solver state cache.
        """
        if x is not None:
            raise ValueError(
                "Cannot use initial estimate of solution for Cholesky. Set `x=None` instead."
            )

        with torch.no_grad():
            # Dense matrix
            K_Winv_dense = K_op.to_dense() + Winv_op.to_dense()

            # Ensure rhs is vector
            rhs = rhs.reshape(-1)
            if x is not None:
                x = x.reshape(-1)

            # ==========================================================================
            # Initial solver state
            # ==========================================================================

            inverse_op = ZeroLinearOperator(
                *K_op.shape, dtype=K_op.dtype, device=K_op.device
            )
            solver_state = LinearSolverState(
                problem=LinearSystem(A=K_op + Winv_op, b=rhs),
                solution=torch.zeros_like(rhs),
                forward_op=None,
                inverse_op=inverse_op,
                residual=rhs,
                residual_norm=torch.linalg.vector_norm(rhs, ord=2),
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
                    "M": None,
                },
            )
            yield solver_state

            # ==========================================================================
            # Final solver state
            # ==========================================================================

            # Inverse operator v -> (K+W^{-1})^{-1}v
            inverse_op = CholeskySolveLinearOperator(K_Winv_dense)

            # Solution
            solution = inverse_op @ rhs

            # Residual
            residual = rhs - K_Winv_dense @ solution

            # Create solver state
            solver_state = LinearSolverState(
                problem=LinearSystem(A=K_op + Winv_op, b=rhs),
                solution=solution,
                forward_op=None,
                inverse_op=inverse_op,
                residual=residual,
                residual_norm=torch.linalg.vector_norm(residual, ord=2),
                logdet=None,
                iteration=K_op.shape[0],
                cache={
                    "search_dir_sq_Anorms": [],
                    "rhs_norm": torch.linalg.vector_norm(rhs, ord=2),
                    "action": None,
                    "observation": None,
                    "search_dir": None,
                    "step_size": None,
                    "actions": None,  # TODO: Could add these if needed downstream.
                    "K_op_actions": None,
                    "M": K_Winv_dense,
                },
            )

            yield solver_state

    def solve(
        self,
        K_op: LinearOperator,
        Winv_op: LinearOperator,
        rhs: Tensor,
        /,
        x: Optional[Tensor] = None,
    ) -> LinearSolverState:
        solver_state = None

        for solver_state in self.solve_iterator(K_op, Winv_op, rhs, x=x):
            pass

        return solver_state
