from __future__ import division
from cvxpy import *
from ncvx import *
import numpy as np

# Chips are assigned to slots in a 1D array.
# Some chips are connected by wires.
# The goal is to minimize the total wiring distance.
n = 10
num_slots = n
num_items = n
num_connections = n-1

# Get connections.
connections = []
np.random.seed(1)
for i in range(num_connections):
    # match = np.random.choice(range(num_items), size=2, replace=False)
    match = (i, i+1)
    smaller = min(match[0], match[1])
    larger = max(match[0], match[1])
    match = (smaller, larger)
    if match not in connections:
        connections.append(match)

print "connections", connections

MAX_ITER = 50
RESTARTS = 5

# Make objective.
assignment = Assign(num_slots, num_items)
positions = assignment.T*np.arange(num_slots)
cost = 0
for chip1, chip2 in connections:
    cost += abs(positions[chip1] - positions[chip2])
prob = Problem(Minimize(cost), [sum_entries(positions) >= n*(n-1)/2])
result = prob.solve(method="NC-ADMM", max_iter=MAX_ITER,
                    restarts=RESTARTS, random=True,
                    solver=ECOS, verbose=False)#, tau=1.1, tau_max=100)
# result = prob.solve(method="relax_and_round")
print "all constraints hold:", np.all([c.value for c in prob.constraints])
print "final value", result
# print prob.solve(method="polish")
# print np.around(positions.value)

assignment = Bool(num_slots, num_items)
positions = assignment.T*np.arange(num_slots)
cost = 0
for chip1, chip2 in connections:
    cost += abs(positions[chip1] - positions[chip2])
prob = Problem(Minimize(cost),
        [assignment*np.ones((n, 1)) == 1,
         np.ones((1, n))*assignment == 1])
prob.solve(solver=GUROBI, verbose=False, TimeLimit=10)
print "gurobi solution", prob.value
# print positions.value

# # Randomly guess permutations.
# total = 0
# best = np.inf
# for k in range(RESTARTS*MAX_ITER):
#     assignment.value = np.zeros(assignment.size)
#     for i, match in enumerate(np.random.permutation(num_slots)):
#         assignment.value[match, i] = 1
#     if cost.value < result:
#         total += 1
#     if cost.value < best:
#         best = cost.value
#     # print positions.value
# print "%% better = ", (total/(RESTARTS*MAX_ITER))
# print "best = ", best
