# Branch and bound on a circle.
from __future__ import division
from cvxpy import *
import numpy as np
try:
    from Queue import PriorityQueue
except:
    from queue import PriorityQueue

# Our approach:
# theta_1 <= tan(x2/x1) <= theta_2
# Find points on chord:
# x_1 = cos(theta_1), x_2 = sin(theta_1) etc.
# Get line between them w = x_{theta_2} - x_{theta_1}
# Rotate -pi/2 degrees to get orthogonal line pointing away from origin.
# q = [[0, 1], [-1, 0]]w (rotation matrix)
# Add constraint q^T(x - x_{theta_1}) >= 0

class Chord(object):
    def __init__(self, var, small_theta, big_theta, radius=1):
        # big_theta >= small_theta
        self.var = var
        self.small_theta = small_theta
        self.big_theta = big_theta
        self.radius = radius

    def hull(self):
        """Returns the convex hull of the chord.
        """
        x_small = np.array([np.cos(self.small_theta),
                            np.sin(self.small_theta)])
        x_big = np.array([np.cos(self.big_theta),
                          np.sin(self.big_theta)])
        w = x_big - x_small
        q = np.matrix("0 1; -1 0").A.dot(w)
        q /= norm(q, 2).value
        return [norm(self.var, 2) <= 1,
                q.T*(self.var - x_small) >= 0]

    def split(self):
        avg_theta = (self.small_theta + self.big_theta)/2
        return [Chord(self.var, self.small_theta, avg_theta, self.radius),
                Chord(self.var, avg_theta, self.big_theta, self.radius)]

    def project(self, val):
        if val is None:
            avg_theta = (self.small_theta + self.big_theta)/2
            return RADIUS*np.array([np.cos(avg_theta),
                                    np.sin(avg_theta)])
        else:
            return RADIUS*x.value/norm(x,2).value

# Branch and bound algorithm.
np.random.seed(1)
x = Variable(2)
w = np.random.randn(2)
cost = norm(x,'inf')
constraints = [abs(x[0]) <= 0.5]

EPS_THETA = 1e-3
EPS_SOL = 1e-3
RADIUS = 1
best_solution = np.inf
best_x = 0
# L = -np.inf
# Used to break ties.
counter = 0
evaluated = 0
# Priority queue.
nodes = PriorityQueue()
nodes.put((-np.inf, counter, Chord(x/RADIUS, 0, 2*np.pi, RADIUS)))
while not nodes.empty():
    # Evaluate the node with the lowest lower bound.
    parent_lower_bound, _, chord = nodes.get()
    # Short circuit if lower bound above upper bound.
    if parent_lower_bound + EPS_SOL >= best_solution:
        continue
    prob = Problem(Minimize(cost), constraints + chord.hull())
    lower_bound = prob.solve()
    evaluated += 1
    # Project onto circle.
    x.value = chord.project(x.value)
    # Get value at projection.
    if np.all([c.value for c in constraints]):
        upper_bound = cost.value
    else:
        upper_bound = np.inf
    # Update upper bound and best x.
    best_solution = min(upper_bound, best_solution)
    if upper_bound <= best_solution:
        best_x = x.value
    # Add new nodes if not a leaf and the branch cannot be pruned.
    # TODO remove pruning here?
    if chord.big_theta - chord.small_theta > EPS_THETA and \
       lower_bound + EPS_SOL < best_solution:
        for sub_chord in chord.split():
            counter += 1
            nodes.put((lower_bound, counter, sub_chord))

sections = 2*np.pi*RADIUS/EPS_THETA
print "Evaluated: %d out of %d, or %.3f%%" % (evaluated,
    sections, 100*evaluated/sections)
print("Optimal solution:", best_solution)
print(best_x)
print "||x||_2 == RADIUS is", np.allclose(norm(best_x).value, RADIUS)
# print(w/norm(w,2).value)
