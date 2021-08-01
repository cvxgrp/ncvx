import cvxpy as cp
import ncvx as nc
import numpy as np
import time

np.random.seed(1)

m = 100; n = 10 * m; k = 10
print(f"m = {m}, n = {n}, k = {k}")

c = np.random.uniform(0, 1, size=(n, 1))
A = np.zeros((m * n, 1))
arr = np.arange(m * n)
np.random.shuffle(arr)
positions = arr[0: m * n // k]
A[positions] = np.random.uniform(0, 5, size=(m * n // k, 1))
A = cp.reshape(A, (m, n)).value
a = np.random.randint(1, 6, size=(n, 1))
z_true = np.zeros((n, 1))
for i in range(n):
    z_true[i] = np.random.randint(0, a[i])
b = A.dot(z_true)

z = nc.Integer((n, 1), M=a)
cost = c.T @ z
constraints = [z >= 0, A @ z <= b]
prob = cp.Problem(cp.Maximize(cost), constraints)


def neighbor_func(z_val, cur_merit):
    # Special function for evaluating neighbors.
    resid = np.dot(A, z_val) - b
    obj = -np.dot(c.T, z_val)
    best_merit = obj + 10000 * np.maximum(resid, 0).sum()
    best_sltn = None
    for i in range(z_val.shape[0]):
        vals = [1, -1] if z_val[i] > 0 else [1]
        for op in vals:
            obj2 = obj - op*c[i]
            resid2 = resid + op*A[:, i]
            rmax = resid2.max()
            if obj2 < best_merit and rmax < 1e-3:
                print(best_merit)
                best_merit = obj2
                best_sltn = i, op
    if best_sltn is not None:
        i, op = best_sltn
        z_val[i] += op
    return best_merit, z_val


# NC-ADMM heuristic
tic = time.perf_counter()
val, resid = prob.solve(method="NC-ADMM", solver=cp.CVXOPT,
                        polish_depth=5, show_progress=True, parallel=False,
                        neighbor_func=neighbor_func, max_iter=25, restarts=5)
toc = time.perf_counter()
print(f"NC-ADMM residual = {resid}")
print(f"NC-ADMM value    = {val}")
print(f"Solve time = {toc - tic:.4f} seconds.")

# Relax-round-polish heuristic
# val, resid = prob.solve(method="relax-round-polish", polish_depth=5)
# print(("Relax-round-polish residual =", resid))
# print(("Relax-round-polish value =", val))

# Global solution via Gurobi.
if cp.GUROBI in cp.installed_solvers():
    z = cp.Variable((n, 1), integer=True)
    cost = c.T * z
    prob = cp.Problem(cp.Maximize(cost),
                      [z <= a, z >= 0, A @ z <= b])
    prob.solve(solver=cp.GUROBI, TimeLimit=20, verbose=True)
    print("Gurobi value =", cost.value)
