from cvxpy import *
from ncvx import *
import numpy as np

np.random.seed(1)

# Generating problem data
p = 1.0 / 10 # each set contains np emelements (10% of the elements) in expectation
m = int(3 / p)  # total number of elements in all sets (with repetition) is mnp ~ 3n
k = int(1/(3*p))  # if you choose k sets randomly they contain knp ~ n/3 element in expectation
n = 100
A = np.random.binomial(1, p, [n, m]);
w = np.random.rand(1,n)

# NC-ADMM heuristic
x = Boolean(n)
y = Choose(m, 1, k)
weight = w * x
constraints = [A * y >= x]
prob = Problem(Maximize(weight), constraints)

prob.solve(method="NC-ADMM")
print "NC-ADMM solution = ", weight.value

# Gurobi (uncomment code below)
x = Bool(n)
y = Bool(m)
weight = w * x
constraints = [sum(y) <= k, A * y >= x]
prob = Problem(Maximize(weight), constraints)

prob.solve(solver = GUROBI)
print "GUROBI solution = ", weight.value
