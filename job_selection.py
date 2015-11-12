from cvxpy import *
from noncvx_admm import *
import numpy as np
np.random.seed(1)

# XXX redo with wide A, z <= a where a in R^n with entries between 1 and 5.
# A, b >= 0. Generate b by multiplying A by a valid, nonzero z.

# Scale m. Put time limit on GUROBI?
# Plot relative error (true - nc-admm)/true
# Plot merit function so relax and round can be included.

m = 10
n = 10*m
k = 10
M = 5
c = np.random.uniform(0, 1, size=(n, 1))
A = np.zeros((m*n,1))
arr = np.arange(m*n)
np.random.shuffle(arr)
positions = arr[0:m*n//k]
A[positions] = np.random.uniform(0, M, size=(m*n//k,1))
A = reshape(A, m, n).value
# b = np.random.uniform(0, k*M, size=(m,1))
a = np.random.randint(1, 6, size=(n, 1))
# z_true = np.random.randint(0, 6, size=(n, 1))
# z_true = np.minimum(a, z_true)
z_true = np.zeros((n,1))
for i in range(n):
    z_true[i] = np.random.randint(0, a[i])
b = A.dot(z_true)

gamma = 1

z = Integer(n)
cost = c.T*z #+ gamma*sum_entries(pos(A*z - b))
prob = Problem(Minimize(-cost), [z <= a, z >= 0, A*z <= b])
val, resid = prob.solve(method="relax_project_polish", gamma=1000, verbose=False)
if resid < 1e-3:
    print "relax and round value", -val
else:
    print "relax and round failed, residual = ", resid


RESTARTS = 10
ITERS = 100
val, resid = prob.solve(method="admm", restarts=RESTARTS,
           rho=np.random.uniform(0,5,size=RESTARTS),
           max_iter=ITERS, random=True, sigma=1, solver=ECOS, gamma=1000)
assert resid < 1e-3
print "ADMM value", -val
# print prob.constraints[0].violation.sum()

z = Int(n)
cost = c.T*z #+ gamma*sum_entries(pos(A*z - b))
prob = Problem(Minimize(-cost), [z <= a, z >= 0, A*z <= b])
prob.solve(solver=GUROBI, verbose=True, TimeLimit=10)
print "true value", cost.value
