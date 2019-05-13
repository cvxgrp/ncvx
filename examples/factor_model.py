from cvxpy import *
from ncvx import *
import numpy as np
import random

np.random.seed(1)
random.seed(1)


# Generating problem data
n = 12; k = n//2
print("n =", n, ", k =", k)
SNR = 20
F = np.random.randn(n, k)
D_true = np.random.exponential(1, size=(n, 1))
Sigma_true = F.dot(F.T) + np.diag(D_true)
variance = norm(Sigma_true, 'fro').value/(np.sqrt(n*n)*SNR)
noise = np.random.normal(0, variance, size=(n,n))
Sigma = Sigma_true + (noise + noise.T)/2


# NC-ADMM heuristic
Sigma_lr = Rank(n, n, k, M=None, symmetric=True)
D_vec = Variable(n); D = diag(D_vec)
cost = sum_squares(Sigma - Sigma_lr - D)
constraints = [D_vec >= 0, Sigma_lr >> 0]
prob = Problem(Minimize(cost), constraints)

def polish_func(sltn):
    matrix = sltn[Sigma_lr.id]
    w, V = np.linalg.eigh(matrix)
    w_sorted_idxs = np.argsort(-w)
    pos_w = w[w_sorted_idxs[:k]]
    pos_V = V[:,w_sorted_idxs[:k]]
    Sigma_tmp = Symmetric(k, k)
    Sigma_small = pos_V*Sigma_tmp*pos_V.T

    D_vec2 = Variable(n); D = diag(D_vec2)
    cost = sum_squares(Sigma - Sigma_small - D)
    constraints = [D_vec >= 0, Sigma_lr >> 0]
    prob = Problem(Minimize(cost), constraints)
    result = prob.solve(solver=SCS)
    return result, {Sigma_lr.id: Sigma_small.value, D_vec.id: D_vec2.value}

prob.solve(method="NC-ADMM", solver=SCS, show_progress=True, parallel=False,
           restarts=1, max_iter=10, polish_func=polish_func, polish_depth=10)
Sigma_lr.value = Sigma_lr.project(Sigma_lr.value)
D_vec.value = pos(D_vec).value
print("NC-ADMM value", cost.value)

w, V = np.linalg.eigh(Sigma_lr.value)
print(w, D_vec.value)
print(sum_squares(Sigma - Sigma_lr - D).value)
print(sum_squares(Sigma - Sigma_lr.project(Sigma)).value)

# Relax-round-polish heuristic
prob.solve(method="relax-round-polish", solver=SCS)
print("Relax-round-polish value", cost.value)

# Nuclear norm heurstic
gamma = Parameter(sign="positive")
Sigma_lr = Semidef(n)
reg = trace(Sigma_lr)
D_vec = Variable(n)
D = diag(D_vec)
cost = sum_squares(Sigma - Sigma_lr - D)
constraints = [D_vec >= 0]
prob = Problem(Minimize(cost + gamma*reg), constraints)
found_lr = False
for gamma_val in np.logspace(-2,2,num=100):
    gamma.value = gamma_val
    prob.solve(solver=SCS)
    w = np.linalg.eigvals(Sigma_lr.value)
    rank = sum(w > 1e-3)
    if rank <= k:
        # Polish.
        polish_prob = Problem(Minimize(cost),
                              constraints + [Sigma_lr == Sigma_lr.value])
        found_lr = True
        break
assert found_lr
print("Nuclear norm value", cost.value)
