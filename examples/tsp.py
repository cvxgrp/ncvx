from cvxpy import *
from ncvx import *
import numpy as np
import matplotlib.pyplot as plt

# Traveling salesman problem.
np.random.seed(1)
n = 60

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



def neighbor_func(Z, cur_merit):
    best_merit = np.dot(D.ravel(), Z.ravel())
    idxs = np.argmax(Z, axis=1)
    best_diff = 0
    for a in range(Z.shape[0]):
        """Swap a->b->c->d to a->c->b->d
        """
        b = idxs[a]
        c = idxs[b]
        d = idxs[c]
        diff = D[a, c] + D[b, d] - D[a, b] - D[c, d]
        if (diff < best_diff):
            best_diff = diff
            a_candid = a
            b_candid = b
            c_candid = c
            d_candid = d

    if (best_diff < 0):
        best_merit += diff
        Z[a_candid, c_candid] = 1
        Z[a_candid, b_candid] = 0

        Z[b_candid, d_candid] = 1
        Z[b_candid, c_candid] = 0

        Z[c_candid, b_candid] = 1
        Z[c_candid, d_candid] = 0
        # print(Z)

    return best_merit, Z







import time
start = time.time()
val, result = prob.solve(method="NC-ADMM", polish_depth=5, solver = SCS,
                         show_progress=True, neighbor_func=neighbor_func, parallel=True, restarts=1,
                         max_iter=50)
end = time.time()
print(end - start)
print("final value", cost.value)

# Plotting
ordered = (X*P.T).value
for i in range(n):
    plt.plot([X[0,i], ordered[0,i]],
             [X[1,i], ordered[1,i]],
             color = 'brown', marker = 'o')

plt.show()
