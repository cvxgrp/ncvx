import cvxpy as cp, numpy as np, cvxopt, matplotlib.pyplot as plt
import ncvx

np.random.seed(1)

N = 41 # number of circles
r = 0.4 + 0.6 * np.random.rand(N) # radii

#define variables.
x_vals = [cp.Variable() for i in range(N)]
y_vals = [cp.Variable() for i in range(N)]

objective = cp.Minimize(cp.maximum( *(x_vals + y_vals) ) + 0.5)
constraints = []
for i in range(N):
    constraints  += [0.5*r[i] <= x_vals[i],
                     0.5*r[i] <= y_vals[i]]
diff_vars = []
for i in range(N-1):
    for j in range(i+1,N):
        t = cp.Variable()
        diff_vars.append(ncvx.Annulus(2, 0.5 * (r[i]+r[j]), N))
        constraints += [
            cp.vstack((x_vals[i] - x_vals[j], y_vals[i] - y_vals[j])) == diff_vars[-1]]

prob = cp.Problem(objective, constraints)
result = prob.solve(method="relax-round-polish", polish_depth=100, verbose=True)

#plot the circles
circ = np.linspace(0,2 * np.pi)
for i in range(N):
    plt.plot(x_vals[i].value+0.5*r[i]*np.cos(circ),y_vals[i].value+0.5*r[i]*np.sin(circ),'b')
plt.xlim([0,objective.value])
plt.ylim([0,objective.value])
plt.axes().set_aspect('equal')
plt.show()
