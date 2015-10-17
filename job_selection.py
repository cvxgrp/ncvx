from cvxpy import *
from noncvx_admm import *
import numpy as np
np.random.seed(1)

n = 100
m = n//2
k = 10
M = 100
c = np.random.randn(n)
A = np.zeros((m*n,1))
arr = np.arange(m*n)
np.random.shuffle(arr)
positions = arr[0:m*n//k]
A[positions] = np.random.uniform(0, M, size=(m*n//k,1))
A = reshape(A, m, n).value
b = np.random.uniform(0, k*M, size=(m,1))

gamma = 1

z = Integer(n)
cost = c.T*z + gamma*sum_entries(pos(A*z - b))
prob = Problem(Minimize(cost), [z <= M, z >= 0])
val, status = prob.solve(method="relax_and_round")
if val < np.inf:
    print "relax and round value", cost.value


RESTARTS = 10
ITERS = 100
prob.solve(method="admm", restarts=RESTARTS,
           rho=np.random.uniform(0,2,size=RESTARTS),
           max_iter=ITERS, random=True, sigma=2, solver=ECOS, gamma=1e5)
print "ADMM value", cost.value
# print prob.constraints[0].violation.sum()

z = Int(n)
cost = c.T*z + gamma*sum_entries(pos(A*z - b))
prob = Problem(Minimize(cost), [z <= M, z >= 0])
prob.solve(solver=GUROBI, verbose=False)
print "true value", cost.value