import cvxpy as cp
import ncvx as nc
import numpy as np

# This example is described in Section 6.5 of the NCVX paper.

rng = np.random.default_rng(seed=123)

# Generating problem data
n = 12
k = n // 2
print(f"n = {n}, k = {k}")
SNR = 20
F = rng.standard_normal(size=(n, k))
D_true = rng.exponential(1, size=(n, 1))
Sigma_true = F @ F.T + np.diag(D_true)
variance = cp.norm(Sigma_true, 'fro').value / (np.sqrt(n * n) * SNR)
noise = rng.normal(0, variance, size=(n, n))
Sigma = Sigma_true + (noise + noise.T) / 2

# NC-ADMM heuristic
Sigma_lr = nc.Rank((n, n), k, M=None, symmetric=True)
D_vec = cp.Variable(n)
D = cp.diag(D_vec)
cost = cp.sum_squares(Sigma - Sigma_lr - D)
constraints = [D_vec >= 0, Sigma_lr >= 0]
prob = cp.Problem(cp.Minimize(cost), constraints)


def polish_func(sltn):
    matrix = sltn[Sigma_lr.id]
    w, V = np.linalg.eigh(matrix)
    w_sorted_idxs = np.argsort(-w)
    pos_w = w[w_sorted_idxs[:k]]
    pos_V = V[:, w_sorted_idxs[:k]]
    Sigma_tmp = cp.Variable((k, k), symmetric=True)
    Sigma_small = pos_V @ Sigma_tmp @ pos_V.T

    D_vec2 = cp.Variable(n); D = cp.diag(D_vec2)
    cost = cp.sum_squares(Sigma - Sigma_small - D)
    constraints = [D_vec >= 0, Sigma_lr >> 0]
    prob = cp.Problem(cp.Minimize(cost), constraints)
    result = prob.solve(solver=cp.SCS)
    return result, {Sigma_lr.id: Sigma_small.value, D_vec.id: D_vec2.value}


prob.solve(method="NC-ADMM", solver=cp.SCS, show_progress=True, parallel=False,
           restarts=1, max_iter=10, polish_func=polish_func, polish_depth=10)
Sigma_lr.value = Sigma_lr.project(Sigma_lr.value)
D_vec.value = cp.pos(D_vec).value
print(f"NC-ADMM value = {cost.value}")

w, V = np.linalg.eigh(Sigma_lr.value)
print(f"w = \n{w}\nD = \n{D_vec.value}")
print(cp.sum_squares(Sigma - Sigma_lr - D).value)
print(cp.sum_squares(Sigma - Sigma_lr.project(Sigma)).value)

# Relax-round-polish heuristic
prob.solve(method="relax-round-polish", solver=cp.SCS)
print(f"Relax-round-polish value = {cost.value}")

# Nuclear norm heuristic
gamma = cp.Parameter(nonneg=True)
Sigma_lr = cp.Variable((n, n), PSD=True)
reg = cp.trace(Sigma_lr)
D_vec = cp.Variable(n)
D = cp.diag(D_vec)
cost = cp.sum_squares(Sigma - Sigma_lr - D)
constraints = [D_vec >= 0]
prob = cp.Problem(cp.Minimize(cost + gamma*reg), constraints)
found_lr = False
for gamma_val in np.logspace(-2, 2, num=100):
    gamma.value = gamma_val
    prob.solve(solver=cp.SCS)
    w = np.linalg.eigvals(Sigma_lr.value)
    rank = sum(w > 1e-3)
    if rank <= k:
        # Polish.
        polish_prob = cp.Problem(cp.Minimize(cost),
                                 constraints + [Sigma_lr == Sigma_lr.value])
        found_lr = True
        break
assert found_lr
print(f"Nuclear norm value = {cost.value}")
