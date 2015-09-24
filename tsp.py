from __future__ import division
from cvxpy import *
from noncvx_admm import *
import numpy as np

# Traveling salesman problem.
n = 10

# Get locations.
np.random.seed(1)
w = 10
# x = np.random.uniform(-w, w, size=(n,1))
# y = np.random.uniform(-w, w, size=(n,1))
x = np.zeros((n,1))
y = np.zeros((n,1))
x[0] = -w + 4*w/n
y[0] = -w
x[1] = x[0]
y[1] = -w + 4*w/n
for i in range(1,n//2):
    x[2*i] = x[2*i-1] + 4*w/n
    y[2*i] = y[2*i-1]
    x[2*i+1] = x[2*i]
    y[2*i+1] = y[2*i] + 4*w/n

MAX_ITER = 1000
RESTARTS = 1

# Make objective.
perm = Assign(n, n)
ordered_x = vstack(-w, perm*x, w)
ordered_y = vstack(-w, perm*y, w)
cost = 0
for i in range(n+1):
    x_diff = ordered_x[i+1] - ordered_x[i]
    y_diff = ordered_y[i+1] - ordered_y[i]
    cost += norm(vstack(x_diff, y_diff))
prob = Problem(Minimize(cost))
result = prob.solve(method="admm", max_iter=MAX_ITER,
                    restarts=RESTARTS, random=True, rho=RESTARTS*[1],
                    solver=ECOS, verbose=False, sigma=10.0, polish=False)#, tau=1.1, tau_max=100)
print "all constraints hold:", np.all([c.value for c in prob.constraints])
print "final value", cost.value

import matplotlib.pyplot as plt
for i in range(n+1):
    plt.plot([ordered_x[i].value, ordered_x[i+1].value],
             [ordered_y[i].value, ordered_y[i+1].value],
             color = 'brown', marker = 'o')
plt.show()

print "relax and round result", prob.solve(method="relax_and_round")

# print prob.solve(method="polish")
# print np.around(positions.value)

# perm = Bool(n, n)
# ordered_x = vstack(-w, perm*x, w)
# ordered_y = vstack(-w, perm*y, w)
# cost = 0
# for i in range(n+1):
#     x_diff = ordered_x[i+1] - ordered_x[i]
#     y_diff = ordered_y[i+1] - ordered_y[i]
#     cost += norm(vstack(x_diff, y_diff))
# prob = Problem(Minimize(cost),
#         [perm*np.ones((n, 1)) == 1,
#          np.ones((1, n))*perm == 1])
# prob.solve(solver=GUROBI, verbose=False, TimeLimit=10)
# print "gurobi solution", prob.value

# for i in range(n+1):
#     plt.plot([ordered_x[i].value, ordered_x[i+1].value],
#              [ordered_y[i].value, ordered_y[i+1].value],
#              color = 'brown', marker = 'o')
# plt.show()
# # print positions.value

# # Randomly guess permutations.
# total = 0
# best = np.inf
# for k in range(RESTARTS*MAX_ITER):
#     perm.value = np.zeros(perm.size)
#     selection = np.random.permutation(n)
#     perm.value[selection, range(n)] = 1
#     val = cost.value
#     if val < result:
#         total += 1
#     if val < best:
#         best = val
#     # print positions.value
# print "%% better = ", (total/(RESTARTS*MAX_ITER))
# print "best = ", best
