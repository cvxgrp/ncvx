from cvxpy import *
from ncvx import *
import numpy as np
import random

np.random.seed(1)
random.seed(1)


# Generating problem data
n = 12; k = n//2
print "n =", n, ", k =", k
SNR = 20
F = np.random.randn(n, k)
D_true = np.random.exponential(1, size=(n, 1))
Sigma_true = F.dot(F.T) + np.diag(D_true)
variance = norm(Sigma_true, 'fro').value/(np.sqrt(n*n)*SNR)
noise = np.random.normal(0, variance, size=(n,n))
Sigma = Sigma_true + noise


# NC-ADMM heuristic
Sigma_lr = Rank(n, n, k, symmetric=True)
D_vec = Variable(n); D = diag(D_vec)
cost = sum_squares(Sigma - Sigma_lr - D)
constraints = [D_vec >= 0, Sigma_lr == Sigma_lr.T, Sigma_lr >> 0]
prob = Problem(Minimize(cost), constraints)
prob.solve(method="NC-ADMM", solver=SCS)
print "NC-ADMM value", cost.value

# Relax-round-polish heuristic
prob.solve(method="relax-round-polish", solver=SCS)
print "Relax-round-polish value", cost.value

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
print "Nuclear norm value", cost.value