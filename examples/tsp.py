from cvxpy import *
from ncvx import *
import numpy as np
import matplotlib.pyplot as plt

# Traveling salesman problem.
np.random.seed(1)
n = 35

# Get locations.
x = np.random.uniform(-1, 1, size=(n,1))
y = np.random.uniform(-1, 1, size=(n,1))
X = np.vstack([x.T,y.T])

# Make distance matrix.
D = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        D[i,j] = norm(X[:,i] - X[:,j]).value


# Approximate solution with NC-ADMM
P = Tour(n)
cost = vec(D).T*vec(P)
prob = Problem(Minimize(cost), [])
result = prob.solve(method="NC-ADMM", solver = SCS)
print "final value", cost.value

# Plotting
ordered = (X*P.T).value
for i in range(n):
    plt.plot([X[0,i], ordered[0,i]],
             [X[1,i], ordered[1,i]],
             color = 'brown', marker = 'o')
plt.show()
