import torch

from linear_operator.linear_solvers import PLS_GPC
from linear_operator.linear_solvers.policies import GradientPolicy
from linear_operator.operators import LowRankRootLinearOperator

from .test_probabilistic_linear_solver_gpc import get_testproblem


def test_compression(top_k=None, kappa=None):
    """With this test function, we can play around with the parameters `top_k` and
    `kappa` and observe how the dummy inputs are compressed.
    """

    torch.manual_seed(0)

    print(f"\ntest_compression: top_k = {top_k}, kappa = {kappa}")

    # Create dummy inputs
    eigvals = torch.rand(5)
    eigvecs = torch.rand(5, 5)
    print("\neigvals = ", eigvals)
    print("eigvecs = \n", eigvecs)

    # Compression
    eigvals_new, eigvecs_new = PLS_GPC.compression(
        eigvals, eigvecs, top_k=top_k, kappa=kappa
    )
    print("\neigvals_new = ", eigvals_new)
    print("eigvecs_new = \n", eigvecs_new)


def test_pls_gpc_with_compression(use_compression=True):
    """Here, we apply the linear solver multiple times on the same test problem. All
    steps use information from the previous step (the actions and K times these
    actions). If `use_compression` is set to `True`, compression is used to keep the
    inverse operator at a constant rank.
    """

    print(f"\ntest_pls_gpc_with_compression: use_compression = {use_compression}")

    # Parameters
    N = 10
    device = torch.device("cpu")
    pls_policy = GradientPolicy()
    iters_per_step = [3, 2, 4, 3, 11]
    top_k = 4 if use_compression else None

    # Create test problem
    K, Winv, rhs, _ = get_testproblem(0, N, device=device)

    # Initialize preconditioner
    actions = None
    K_op_actions = None

    # Count number of iterations in total
    iters_counter = 0

    for step_idx, pls_max_iters in enumerate(iters_per_step):

        print(f"\nStep {step_idx}")
        pls = PLS_GPC(policy=pls_policy, max_iter=pls_max_iters)

        with torch.no_grad():
            solve_iterator = pls.solve_iterator(
                K,
                Winv,
                rhs,
                actions=actions,
                K_op_actions=K_op_actions,
                top_k=top_k,
            )
            for solver_state in solve_iterator:
                print("  Iteration", solver_state.iteration)

                # Increase iterations counter
                if solver_state.iteration != 0:
                    iters_counter += 1

                # Print info about the preconditioner
                if solver_state.cache["actions"] is not None:
                    actions_shape = tuple(solver_state.cache["actions"].shape)
                    print(f"    actions shape = {actions_shape}")
                if isinstance(solver_state.inverse_op, LowRankRootLinearOperator):
                    root_shape = tuple(solver_state.inverse_op.root.shape)
                    print(f"    root shape    = {root_shape}")

        # Termination criterion
        if solver_state.residual_norm <= 1e-5:
            print(f"\nTermination crit. fulfilled after {iters_counter} iterations\n")
            break

        # Extract preconditioner
        actions = solver_state.cache["actions"]
        K_op_actions = solver_state.cache["K_op_actions"]


if __name__ == "__main__":
    test_compression(top_k=None, kappa=None)
    test_compression(top_k=2, kappa=None)
    test_compression(top_k=None, kappa=0.5)
    test_compression(top_k=2, kappa=0.5)

    test_pls_gpc_with_compression(use_compression=False)
    test_pls_gpc_with_compression(use_compression=True)
