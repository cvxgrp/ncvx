from cvxpy import *
from ncvx import *
import numpy as np
import random

np.random.seed(1)
random.seed(1)

# Generating problem data
m = 65; n = 2*m; k = n//10
print "m =", m, ", n =", n, ", k =", k
SNR = 20; M = 1
A = np.random.randn(m, n)
x_true = np.random.uniform(-1, 1, size=(n,1))
zeros = random.sample(range(n), n-k)
x_true[zeros] = 0
sigma = norm(A.dot(x_true), 2).value/(np.sqrt(n)*SNR)
noise = np.random.normal(0, sigma, size=(m,1))
b = A.dot(x_true) + noise


# Best solution using Gurobi after 1 minute. (Uncomment the code below.)
#x = Variable(n)
#bound = Bool(n)
#constr = [abs(x) <= M*bound, sum_entries(bound) <= k]
#cost = sum_squares(A*x - b)
#prob = Problem(Minimize(cost), constr)
#prob.solve(solver=GUROBI, TimeLimit = 60)
#print "Gurobi value =", cost.value
#print "Gurobi solution =\n", np.around(x.value.T, decimals=3)
#print "--------------------------------------------------------------------"

# NC-ADMM heuristic.
x = Card(n, k, M)
cost = sum_squares(A*x - b)
prob = Problem(Minimize(cost))
RESTARTS = 5; ITERS = 50
prob.solve(method="NC-ADMM", restarts=RESTARTS, num_procs = 5, max_iter=ITERS)
print "NC-ADMM value =", cost.value
print "NC-ADMM solution =\n", np.around(x.value.T, decimals=3)
print "--------------------------------------------------------------------"

# Relax-round-polish heurisitc
prob.solve(method="relax_round_polish")
print "Relax-round-polish value =", cost.value
print "Relax-round-polish solution =\n", np.around(x.value.T, decimals=3)
print "--------------------------------------------------------------------"

# Lasso heuristic
gamma = Parameter(sign="positive")
reg = norm(x,1)
cost = sum_squares(A*x - b)
prob = Problem(Minimize(cost + gamma*reg), [abs(x) <= M])
found_card = False
for gamma_val in np.logspace(-2,4,num=100):
    gamma.value = gamma_val
    prob.solve()
    card = sum(abs(x).value > 1e-3)
    if card <= k:
        # Polish.
        polish_prob = Problem(Minimize(cost),
                              [abs(x) <= (abs(x).value > 1e-3)*M])
        polish_prob.solve()
        found_card = True
        break
assert found_card
print "Lasso value =", cost.value
print "Lasso solution =\n", np.around(x.value.T, decimals=3)


