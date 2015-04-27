from cvxpy import *
from noncvx_admm import *
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
    #match = np.random.choice(range(num_items), size=2, replace=False)
    match = (i, i+1)
    smaller = min(match[0], match[1])
    larger = max(match[0], match[1])
    match = (smaller, larger)
    if match not in connections:
        connections.append(match)

print "connections", connections

# Make objective.
assignment = Assign(num_slots, num_items)
positions = assignment.T*np.arange(num_slots)
cost = 0
for chip1, chip2 in connections:
    cost += abs(positions[chip1] - positions[chip2])
prob = Problem(Minimize(cost))
result = prob.solve(method="consensus", max_iter=100,
                    restarts=1, random=False, rho=[10])
print result
# print prob.solve(method="polish")
print np.around(positions.value)

for k in range(10):
    assignment.value = np.zeros(assignment.size)
    for i, match in enumerate(np.random.permutation(num_slots)):
        assignment.value[match, i] = 1
    print cost.value
    # print positions.value