from cvxpy import *
from ncvx import *
import numpy as np
np.random.seed(1)

# XXX redo with wide A, z <= a where a in R^n with entries between 1 and 5.
# A, b >= 0. Generate b by multiplying A by a valid, nonzero z.

# Scale m. Put time limit on GUROBI?
# Plot relative error (true - nc-admm)/true
# Plot merit function so relax and round can be included.

m = 20
n = 5*m
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

z = Int(n)
cost = c.T*z #+ gamma*sum_entries(pos(A*z - b))
prob = Problem(Minimize(-cost), [z <= a, z >= 0, A*z <= b])
# prob.solve(solver=GUROBI, verbose=True, TimeLimit=10)
print "true value", cost.value

z = Integer(n)
penalty = 0
# z = 0
# for i in range(3):
#     bool_z = Annulus(n, np.sqrt(n), n)
#     z += (2**i)*(bool_z+1)/2
#     penalty += 1000*(norm(bool_z, 'inf') - 1)

cost = c.T*z #+ gamma*sum_entries(pos(A*z - b))
prob = Problem(Minimize(-cost + penalty), [z <= a, z >= 0, A*z <= b])
# val, resid = prob.solve(method="relax_round_polish", gamma=1000, verbose=False)
# if resid < 1e-3:
#     print "relax and round value", -val
# else:
#     print "relax and round failed, residual = ", resid


RESTARTS = 5
ITERS = 50
val, resid = prob.solve(method="NC-ADMM", restarts=RESTARTS, show_progress=True,
           rho=np.random.uniform(0,1,size=RESTARTS), polish_best=False,
           # rho=RESTARTS*[0],
           num_proj=10, parallel=True, prox_polished=False,
           max_iter=ITERS, random=True, sigma=1, solver=ECOS, gamma=1e6)
print "ADMM value", cost.value
print "ADMM resid", resid
assert resid < 1e-3
print norm(z - np.round(z.value)).value
# print prob.constraints[0].violation.sum()

