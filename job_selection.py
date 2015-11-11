from cvxpy import *
from noncvx_admm import *
import numpy as np
np.random.seed(1)

# XXX redo with wide A, z <= a where a in R^n with entries between 1 and 5.
# A, b >= 0. Generate b by multiplying A by a valid, nonzero z.

# Scale m. Put time limit on GUROBI?
# Plot relative error (true - nc-admm)/true
# Plot merit function so relax and round can be included.

m = 45
n = 10*m
k = 10
M = 5
c = np.random.randn(n)
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
val, status = prob.solve(method="relax_project_polish")
if val < np.inf:
    print "relax and round value", -val
else:
    print "relax and round failed"


RESTARTS = 10
ITERS = 50
prob.solve(method="admm", restarts=RESTARTS,
           rho=np.random.uniform(0,1,size=RESTARTS),
           max_iter=ITERS, random=True, sigma=1, solver=ECOS, gamma=1e5)
print "ADMM value", cost.value
# print prob.constraints[0].violation.sum()

z = Int(n)
cost = c.T*z #+ gamma*sum_entries(pos(A*z - b))
prob = Problem(Minimize(-cost), [z <= a, z >= 0, A*z <= b])
prob.solve(solver=GUROBI, verbose=True)
print "true value", cost.value