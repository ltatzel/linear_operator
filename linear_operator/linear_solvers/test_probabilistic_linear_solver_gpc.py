import pytest
import torch

from linear_operator.linear_solvers import PLS_GPC
from linear_operator.linear_solvers.policies import (
    GradientPolicy,
    UnitVectorPolicy,
)

ATOL = 1e-4
RTOL = 1e-3


def allclose(A, B):
    return torch.allclose(A, B, atol=ATOL, rtol=RTOL)


def get_testproblem(seed, N, device):
    """Construct a testproblem with a linear system `(K + Winv) @ sol = rhs` of size
    `N x N`. We make sure that `K + W_inv` is positive definite.
    """

    torch.manual_seed(seed)

    # Operators `K` and `Winv`
    K = 2 * torch.rand(N, N) - 1
    K = K @ K.T + 1e-4 * torch.eye(N)
    K = K.to(device)
    Winv = torch.diag(torch.rand(N) + 1e-4).to(device)
    assert torch.all(torch.linalg.eigvalsh(K + Winv) > 0), "K + Winv not pos. definite"
    assert torch.allclose(K + Winv, (K + Winv).T), "K + Winv not symmetric"

    # Sample solution, construct rhs
    sol = 2 * torch.rand(N, device=device) - 1
    rhs = (K + Winv) @ sol

    return K, Winv, rhs, sol


def run_solver(
        pls_policy,
        pls_max_iter,
        K,
        Winv,
        rhs,
        x=None,
        actions=None,
        K_op_actions=None,
        top_k=None,
        kappa=None,
):
    """Run the `PLS_GPC` solver for `pls_max_iter` iterations. Return the final solver
    state.
    """

    pls = PLS_GPC(policy=pls_policy, max_iter=pls_max_iter)

    with torch.no_grad():
        solve_iterator = pls.solve_iterator(
            K,
            Winv,
            rhs,
            x=x,
            actions=actions,
            K_op_actions=K_op_actions,
            top_k=top_k,
            kappa=kappa,
        )
        for solver_state in solve_iterator:
            pass

    return solver_state


def check_consistency(solver_state):
    """Test the consistency of solution, residual, inverse approximation and actions."""
    lin_op = solver_state.problem.A
    rhs = solver_state.problem.b
    solution = solver_state.solution
    assert allclose(solution, solver_state.inverse_op @ rhs)
    assert allclose(rhs - lin_op @ solution, solver_state.residual)

    if solver_state.cache["actions"] is not None:
        S_i = solver_state.cache["actions"]
    C_i = S_i @ torch.linalg.solve(S_i.T @ lin_op @ S_i, S_i.T)
    assert torch.allclose(C_i, solver_state.inverse_op.to_dense(), atol=1e-6)


# Define test cases
DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))
DEVICES_IDS = [f"device = {str(d)}" for d in DEVICES]

SEEDS = [0, 1, 42]
SEEDS_IDS = [f"seed = {s}" for s in SEEDS]

NS = [4, 10]
NS_IDS = [f"N = {N}" for N in NS]

PLS_POLICIES = [GradientPolicy(), UnitVectorPolicy()]
PLS_POLICIES_IDS = ["Policy = CG", "Policy = Cholesky"]

PRE_PLS_POLICIES = [GradientPolicy(), UnitVectorPolicy()]
PRE_PLS_POLICIES_IDS = ["PrePolicy = CG", "PrePolicy = Cholesky"]

TOP_K = [None, 3]
TOP_K_IDS = [str(top_k) for top_k in TOP_K]

KAPPAS = [None, 0.5]
KAPPAS_IDS = [str(kappa) for kappa in KAPPAS]


@pytest.mark.parametrize("seed", SEEDS, ids=SEEDS_IDS)
@pytest.mark.parametrize("N", NS, ids=NS_IDS)
@pytest.mark.parametrize("pls_policy", PLS_POLICIES, ids=PLS_POLICIES_IDS)
@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
def test_preconditioner(seed, N, pls_policy, device):
    """Perform some sanity checks on the preconditioner, i.e. on the `actions` and
    `K_op_actions`.
    """

    K, Winv, rhs, _ = get_testproblem(seed, N, device)

    # Run preconditioner for `pre_iters` iterations
    pre_iters = N//2

    # Construct preconditioner
    pre_solver_state = run_solver(pls_policy, pre_iters, K, Winv, rhs)
    actions = pre_solver_state.cache["actions"]
    K_op_actions = pre_solver_state.cache["K_op_actions"]

    # Perform some tests on the preconditioner
    assert actions.shape == torch.Size([N, pre_iters])
    assert K_op_actions.shape == torch.Size([N, pre_iters])
    assert allclose(K_op_actions, K @ actions)

    # Check consistency of solution, residual and inverse approximation
    check_consistency(pre_solver_state)


@pytest.mark.parametrize("seed", SEEDS, ids=SEEDS_IDS)
@pytest.mark.parametrize("N", NS, ids=NS_IDS)
@pytest.mark.parametrize("pls_policy", PLS_POLICIES, ids=PLS_POLICIES_IDS)
@pytest.mark.parametrize("top_k", TOP_K, ids=TOP_K_IDS)
@pytest.mark.parametrize("kappa", KAPPAS, ids=KAPPAS_IDS)
@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
def test_initial_state_of_preconditioned_solver(
    seed, N, pls_policy, top_k, kappa, device
):
    """Here, we test the initial state of the solver when a preconditioner (i.e.
    `actions` and `K_op_actions`) is used.
    """

    K, Winv, rhs, _ = get_testproblem(seed, N, device)

    # Run preconditioner for `pre_iters` iterations
    pre_iters = N//2
    pre_solver_state = run_solver(pls_policy, pre_iters, K, Winv, rhs)
    actions = pre_solver_state.cache["actions"]
    K_op_actions = pre_solver_state.cache["K_op_actions"]

    # Start another solver run using the preconditioner
    pls_max_iter = 0  # <-- test initial state of solver
    solver_state = run_solver(
        pls_policy,
        pls_max_iter,
        K,
        Winv,
        rhs,
        actions=actions,
        K_op_actions=K_op_actions,
        top_k=top_k,
        kappa=kappa,
    )

    # Check consistency of solution, residual, actions and inverse approximation
    check_consistency(solver_state)

    if top_k is None and kappa is None:  # no compression
        
        # Construct root of C_i by hand and compare to the one in `solver_state`
        eigenvals, U = torch.linalg.eigh(actions.T @ (K + Winv) @ actions)
        root = actions @ U @ torch.diag(torch.sqrt(1 / eigenvals))
        assert allclose(root, solver_state.inverse_op.root.to_dense())

        # Construct C_i by hand and compare to the one in `solver_state`
        inverse_op_inner = torch.linalg.inv(actions.T @ (K + Winv) @ actions)
        inverse_op = actions @ inverse_op_inner @ actions.T
        assert allclose(inverse_op, solver_state.inverse_op.to_dense())

        # Check shape of actions after second run
        num_actions_total = pre_solver_state.iteration + solver_state.iteration
        assert solver_state.cache["actions"].shape[1] == num_actions_total
        assert solver_state.cache["K_op_actions"].shape[1] == num_actions_total


@pytest.mark.parametrize("seed", SEEDS, ids=SEEDS_IDS)
@pytest.mark.parametrize("N", NS, ids=NS_IDS)
@pytest.mark.parametrize("pls_policy", PLS_POLICIES, ids=PLS_POLICIES_IDS)
@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
def test_is_solved_without_preconditioner(seed, N, pls_policy, device):
    """Test if the linear system is actually solved when no preconditioner is used."""

    K, Winv, rhs, solution_ref = get_testproblem(seed, N, device)
    pls_max_iter = N

    # Case 1: `x is None`
    solver_state_1 = run_solver(pls_policy, pls_max_iter, K, Winv, rhs, x=None)

    # Case 2: `x` is set to a random initialization
    x = (2 * torch.rand(N) - 1).to(device)
    solver_state_2 = run_solver(pls_policy, pls_max_iter, K, Winv, rhs, x=x)

    # Check both solutions
    for solver_state in [solver_state_1, solver_state_2]:
        assert allclose(solver_state.solution, solution_ref)


@pytest.mark.parametrize("seed", SEEDS, ids=SEEDS_IDS)
@pytest.mark.parametrize("N", NS, ids=NS_IDS)
@pytest.mark.parametrize("pre_pls_policy", PRE_PLS_POLICIES, ids=PRE_PLS_POLICIES_IDS)
@pytest.mark.parametrize("device", DEVICES, ids=DEVICES_IDS)
def test_is_solved_cg_with_preconditioner(seed, N, pre_pls_policy, device):
    """Test if the linear system is actually solved with CG actions when a
    preconditioner (partial CG or partial Cholesky) is used.
    """

    # Create test problem
    K, Winv, rhs, solution_ref = get_testproblem(seed, N, device)

    # Construct preconditioner
    pre_iters = N//2
    pre_solver_state = run_solver(pre_pls_policy, pre_iters, K, Winv, rhs)

    # Run solver a second time on the same problem
    pls_max_iter = N
    pls_policy = GradientPolicy()
    solver_state = run_solver(
        pls_policy,
        pls_max_iter,
        K,
        Winv,
        rhs,
        x=None,
        actions=pre_solver_state.cache["actions"],
        K_op_actions=pre_solver_state.cache["K_op_actions"],
    )

    # Check solution
    assert allclose(solver_state.solution, solution_ref)
    assert solver_state.iteration < N  # Preconditioning "helped"


if __name__ == "__main__":
    test_initial_state_of_preconditioned_solver(
        seed=0, N=4, pls_policy=GradientPolicy(), top_k=2, kappa=None, device="cpu"
    )
