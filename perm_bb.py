# Branch and bound on permutation.
from __future__ import division
from cvxpy import *
from noncvx_admm import *
from noncvx_admm.admm_problem import polish
import numpy as np
from scipy.misc import factorial
try:
    from Queue import PriorityQueue
except:
    from queue import PriorityQueue

def polish(prob, *args, **kwargs):
    # Fix noncvx variables and solve.
    fix_constr = []
    for var in prob.variables():
        if getattr(var, "noncvx", False):
            z = var.project(var.value)
            fix_constr += var.fix(np.matrix(z))
    prob = Problem(prob.objective, prob.constraints + fix_constr)
    prob.solve(*args, **kwargs)
    return prob.value, prob.status

# Branch and bound algorithm.
n = 10

# Get locations.
np.random.seed(1)
w = 10
x = np.random.uniform(-w, w, size=(n,1))
y = np.random.uniform(-w, w, size=(n,1))
# x = np.zeros((n,1))
# y = np.zeros((n,1))
# x[0] = -w + 4*w/n
# y[0] = -w
# x[1] = x[0]
# y[1] = -w + 4*w/n
# for i in range(1,n//2):
#     x[2*i] = x[2*i-1] + 4*w/n
#     y[2*i] = y[2*i-1]
#     x[2*i+1] = x[2*i]
#     y[2*i+1] = y[2*i] + 4*w/n
X = np.vstack([x.T,y.T])

def get_prob(X, w, choices):
    # Make perm with restricted choices.
    k = len(choices)
    if k > 0:
        top = np.zeros((k,n))
        for i in range(k):
            top[i,choices[i]] = 1
        tmp = Assign(n-k, n-k)
        counter = 0
        mats = []
        for i in range(n):
            if i in choices:
                mats += [np.zeros((n-k, 1))]
            else:
                mats += [tmp[:,counter]]
                counter += 1
        perm = vstack(top, hstack(*mats))
    # elif k == n-1:
    #     perm = np.zeros((n,n))
    #     for i in range(k):
    #         perm[i,choices[i]] = 1
    #     for i in range(n):
    #         if i not in choices:
    #             perm[n-1,i] = 1
    else:
        perm = Assign(n, n)

    ordered = hstack([-w,-w],
                     X*perm,
                     [w,w])
    cost = 0
    for i in range(n+1):
        cost += norm(ordered[:,i+1] - ordered[:,i])
    return Problem(Minimize(cost)), perm

EPS_SOL = 1e-3
RADIUS = 1
best_solution = np.inf
best_perm = 0
# L = -np.inf
# Used to break ties.
counter = 0
evaluated = 0
# Priority queue.
nodes = PriorityQueue()
nodes.put((-np.inf, counter, []))
while not nodes.empty():
    # Evaluate the node with the lowest lower bound.
    parent_lower_bound, _, choices = nodes.get()
    # Short circuit if lower bound above upper bound.
    if parent_lower_bound + EPS_SOL >= best_solution:
        continue
    prob, perm = get_prob(X, w, choices)
    lower_bound = prob.solve()
    evaluated += 1
    upper_bound, status = polish(prob)
    # Update upper bound and best x.
    best_solution = min(upper_bound, best_solution)
    if upper_bound <= best_solution:
        print best_solution, lower_bound
        best_perm = perm.value
    # Add new nodes if not a leaf and the branch cannot be pruned.
    # TODO remove pruning here?
    if len(choices) < n and \
       lower_bound + EPS_SOL < best_solution:
        for i in range(n):
            if i not in choices:
                counter += 1
                # k = len(choices)
                # nodes.put((lower_bound, -perm[k,i].value, choices + [i]))
                nodes.put((lower_bound, np.random.random(), choices + [i]))

sections = factorial(n)
print "Evaluated: %d out of %d, or %.3f%%" % (evaluated,
    sections, 100*evaluated/sections)
print("Optimal solution:", best_solution)

import matplotlib.pyplot as plt
ordered = hstack([-w,-w],
                 X*best_perm,
                 [w,w])
for i in range(n+1):
    plt.plot(ordered[0,i:i+2].value.T,
             ordered[1,i:i+2].value.T,
             color = 'brown', marker = 'o')
plt.show()
