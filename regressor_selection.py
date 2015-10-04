from __future__ import division
from cvxpy import *
from noncvx_admm import *
import numpy as np

np.random.seed(1)

m = 40
n = 30
A = np.random.randn(m, n)
b = np.random.randn(m, 1)
k = int(np.floor(n/10))
M = 100

# True solution.
x = Variable(n)
bound = Bool(n)
constr = [abs(x) <= M*bound, sum_entries(bound) <= k]
cost = norm(A*x - b)
prob = Problem(Minimize(cost), constr)
prob.solve(solver=GUROBI)
print "true value", prob.value
# print np.around(x.value, decimals=3)

# ADMM solution.
x = Card(n, k, M)
cost = norm(A*x - b)
prob = Problem(Minimize(cost))

RESTARTS = 1
ITERS = 50
prob.solve(method="admm", restarts=RESTARTS,
           max_iter=ITERS, solver=ECOS, random=True)
print "ADMM value", prob.value
# print np.around(x.value, decimals=3)

print "relax and round value", prob.solve(method="relax_and_round")

# Compare to LASSO.