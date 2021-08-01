import cvxpy as cp
import ncvx as nc
import numpy as np
import random
from multiprocessing import freeze_support

# This example is described in Section 6.1 of the NCVX paper.


def run():
    np.random.seed(1)
    random.seed(1)

    # Generating problem data
    m = 65
    n = 2 * m
    k = n // 10
    print(f"m = {m}, n = {n}, k = {k}")
    SNR = 20
    M = 1
    A = np.random.randn(m, n)
    x_true = np.random.uniform(-1, 1, size=(n, 1))
    zeros = random.sample(list(range(n)), n - k)
    x_true[zeros] = 0
    sigma = cp.norm(A.dot(x_true), 2).value / (np.sqrt(n) * SNR)
    noise = np.random.normal(0, sigma, size=(m, 1))
    b = A.dot(x_true) + noise

    # Best solution using Gurobi after 1 minute.
    if cp.GUROBI in cp.installed_solvers():
        x = cp.Variable(n)
        bound = cp.Variable(n, boolean=True)
        constr = [cp.abs(x) <= M * bound, cp.sum(bound) <= k]
        cost = cp.sum_squares(A @ x - b)
        prob = cp.Problem(cp.Minimize(cost), constr)
        prob.solve(solver=cp.GUROBI, TimeLimit=60)
        print("Gurobi value =", cost.value)
        print("Gurobi solution =\n", np.around(x.value.T, decimals=3))
        print("--------------------------------------------------------------------")

    # NC-ADMM heuristic.
    x = nc.Card(n, k, M)
    cost = cp.sum_squares(A @ x - b)
    prob = cp.Problem(cp.Minimize(cost))
    RESTARTS = 5;
    ITERS = 50
    prob.solve(method="NC-ADMM", restarts=RESTARTS,
               num_procs=5, max_iter=ITERS)
    print("NC-ADMM value =", cost.value)
    print("NC-ADMM solution =\n", np.around(x.value.T, decimals=3))
    print("--------------------------------------------------------------------")

    # Relax-round-polish heuristic
    prob.solve(method="relax-round-polish")
    print("Relax-round-polish value =", cost.value)
    print("Relax-round-polish solution =\n", np.around(x.value.T, decimals=3))
    print("--------------------------------------------------------------------")

    # Lasso heuristic
    gamma = cp.Parameter(nonneg=True)
    reg = cp.norm(x, 1)
    cost = cp.sum_squares(A * x - b)
    prob = cp.Problem(cp.Minimize(cost + gamma * reg), [cp.abs(x) <= M])
    found_card = False
    for gamma_val in np.logspace(-2, 4, num=100):
        gamma.value = gamma_val
        prob.solve()
        card = sum(cp.abs(x).value > 1e-3)
        if card <= k:
            # Polish.
            polish_prob = cp.Problem(cp.Minimize(cost),
                                     [cp.abs(x) <= (cp.abs(x).value > 1e-3) * M])
            polish_prob.solve()
            found_card = True
            break
    assert found_card
    print("Lasso value =", cost.value)
    print("Lasso solution =\n", np.around(x.value.T, decimals=3))


if __name__ == '__main__':
    freeze_support()
    run()
