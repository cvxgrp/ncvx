import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import ncvx as nc

# This example is described in Section 6.3 of the NCVX paper.

rng = np.random.default_rng(seed=123)

N = 41                         # number of circles
r = 0.2 + 0.3 * rng.random(N)  # radii

# Define variables.
x_vals = [cp.Variable() for i in range(N)]
y_vals = [cp.Variable() for i in range(N)]

objective = cp.Minimize(cp.maximum(*(x_vals + y_vals)) + 0.5)
constraints = []
for i in range(N):
    constraints += [r[i] <= x_vals[i],
                    r[i] <= y_vals[i]]
diff_vars = []
for i in range(N - 1):
    for j in range(i + 1, N):
        t = cp.Variable()
        diff_vars.append(nc.Annulus(2, r[i] + r[j], N))
        constraints += [
             cp.vstack((x_vals[i] - x_vals[j], y_vals[i] - y_vals[j])) == diff_vars[-1]]

prob = cp.Problem(objective, constraints)
result = prob.solve(method="relax-round-polish", polish_depth=100)

# Plot the circles.
circ = np.linspace(0, 2 * np.pi)
for i in range(N):
    plt.plot(x_vals[i].value + r[i] * np.cos(circ), y_vals[i].value + r[i] * np.sin(circ), 'b')
plt.xlim([0, objective.value])
plt.ylim([0, objective.value])
plt.axes().set_aspect('equal')
plt.show()
