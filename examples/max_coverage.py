import cvxpy as cp
import ncvx as nc
import numpy as np
import time

np.random.seed(1)

# Generating problem data
p = 1.0 / 10  # each set contains np elements (10% of the elements) in expectation
m = int(3 / p)  # total number of elements in all sets (with repetition) is mnp ~ 3n
k = int(
    1 / (3 * p)
)  # if you choose k sets randomly they contain knp ~ n/3 element in expectation
n = 100
A = np.random.binomial(1, p, [n, m])
w = np.random.rand(1, n)

# NC-ADMM heuristic
x = nc.Boolean((n, 1))
y = nc.Choose((m, 1), k)
weight = w * x
constraints = [A @ y >= x]
prob = cp.Problem(cp.Maximize(weight), constraints)

tic = time.perf_counter()
prob.solve(method="NC-ADMM", parallel=False, verbose=True, polish_depth=0, restarts=1)
toc = time.perf_counter()
print(f"NC-ADMM solution = {weight.value}")
print(f"Solve time = {toc - tic:.4f} seconds.")

# Gurobi.
if cp.GUROBI in cp.installed_solvers():
    x = cp.Variable(n, boolean=True)
    y = cp.Variablel(m, boolean=True)
    weight = w * x
    constraints = [sum(y) <= k, A @ y >= x]
    prob = cp.Problem(cp.Maximize(weight), constraints)
    prob.solve(solver=cp.GUROBI, verbose=True)
    print(f"GUROBI solution = {weight.value}")
