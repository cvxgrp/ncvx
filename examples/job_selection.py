from cvxpy import *
from ncvx import *
import numpy as np

np.random.seed(1)

m = 20; n = 10*m; k = 10
print "m =", m, ", n =", n, ", k =", k

c = np.random.uniform(0, 1, size=(n, 1))
A = np.zeros((m*n,1))
arr = np.arange(m*n)
np.random.shuffle(arr)
positions = arr[0:m*n//k]
A[positions] = np.random.uniform(0, 5, size=(m*n//k,1))
A = reshape(A, m, n).value
a = np.random.randint(1, 6, size=(n, 1))
z_true = np.zeros((n,1))
for i in range(n):
    z_true[i] = np.random.randint(0, a[i])
b = A.dot(z_true)


# NC-ADMM heuristic
z = Integer(n)
cost = c.T*z
constraints = [z <= a, z >= 0, A*z <= b]
prob = Problem(Minimize(-cost), constraints)
val, resid = prob.solve(method="NC-ADMM")
print "NC-ADMM residual =", resid
print "NC-ADMM value =", -val 

# Relax-round-polish heuristic
val, resid = prob.solve(method="relax-round-polish")
print "Relax-round-polish residual =", resid
print "Relax-round-polish value =", -val

## Global solution via Gurobi. (Uncooment the code below.)
z = Int(n)
cost = c.T*z 
prob = Problem(Maximize(cost),
               [z <= a, z >= 0, A*z <= b])
prob.solve(solver=GUROBI, TimeLimit = 20)
print "Gurobi value =", cost.value
