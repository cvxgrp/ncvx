from __future__ import division
from cvxpy import *
from noncvx_admm import *
import numpy as np
import random

np.random.seed(1)
random.seed(1)

n = 10
k = 5
SNR = 20
SCALE = 1
F = np.random.randn(n, k)*SCALE
D_true = np.random.chisquare(1, size=(n, 1))*SCALE
Sigma_true = F.dot(F.T) + np.diag(D_true)


variance = norm(Sigma_true, 2).value/(np.sqrt(n*n)*SNR)
noise = np.random.normal(0, variance, size=(n,n))
Sigma = Sigma_true + noise

print "Sigma true cost", sum_squares(noise).value

# # True solution.
# x = Variable(n)
# bound = Bool(n)
# constr = [abs(x) <= M*bound, sum_entries(bound) <= k]
# cost = sum_squares(A*x - b)
# prob = Problem(Minimize(cost), constr)
# prob.solve(solver=GUROBI)
# print "true value", cost.value
# # print np.around(x.value, decimals=3)

Sigma_lr = Rank(n, n, k, symmetric=True)
D_vec = Variable(n)
D = diag(D_vec)
cost = sum_squares(Sigma - Sigma_lr - D)
constraints = [D_vec >= 0, Sigma_lr == Sigma_lr.T, Sigma_lr >> 0]
prob = Problem(Minimize(cost), constraints)
prob.solve(method="relax_and_round", solver=SCS)
print "relax and round value", cost.value

# ADMM solution.
prob = Problem(Minimize(cost), constraints)
RESTARTS = 5
ITERS = 50
prob.solve(method="admm", restarts=RESTARTS,
           # rho=np.random.uniform(1,5,size=RESTARTS),
           max_iter=ITERS, solver=SCS, random=True, sigma=1)
print "ADMM value", cost.value

# Compare to nuclear norm.
gamma = Parameter(sign="positive")
Sigma_lr = Semidef(n)
reg = trace(Sigma_lr)
D_vec = Variable(n)
D = diag(D_vec)
cost = sum_squares(Sigma - Sigma_lr - D)
constraints = [D_vec >= 0]
prob = Problem(Minimize(cost + gamma*reg), constraints)
found_lr = False
for gamma_val in np.logspace(-2,2,num=250):
    gamma.value = gamma_val
    prob.solve(solver=SCS)
    w = np.linalg.eigvals(Sigma_lr.value)
    rank = sum(w > 1e-3)
    if rank <= k:
        print "rank = ", rank
        # Polish.
        polish_prob = Problem(Minimize(cost),
                              constraints + [Sigma_lr == Sigma_lr.value])
        print polish_prob.solve(solver=SCS)
        found_lr = True
        break
assert found_lr
print "nuclear norm value", cost.value

