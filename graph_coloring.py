from __future__ import division
from cvxpy import *
from ncvx import *
import networkx as nx
from networkx.algorithms.approximation.clique import max_clique as max_clique_approx
import random
import numpy as np
from matplotlib import colors

np.random.seed(1)
random.seed(1)

kelly_colors_hex = [
    "#FFB300", # Vivid Yellow
    "#803E75", # Strong Purple
    "#FF6800", # Vivid Orange
    "#A6BDD7", # Very Light Blue
    "#C10020", # Vivid Red
    "#CEA262", # Grayish Yellow
    "#817066", # Medium Gray

    # The following don't work well for people with defective color vision
    "#007D34", # Vivid Green
    "#F6768E", # Strong Purplish Pink
    "#00538A", # Strong Blue
    "#FF7A5C", # Strong Yellowish Pink
    "#53377A", # Strong Violet
    "#FF8E00", # Vivid Orange Yellow
    "#B32851", # Strong Purplish Red
    "#F4C800", # Vivid Greenish Yellow
    "#7F180D", # Strong Reddish Brown
    "#93AA00", # Vivid Yellowish Green
    "#593315", # Deep Yellowish Brown
    "#F13A13", # Vivid Reddish Orange
    "#232C16", # Dark Olive Green
    ]

n = 25
m = (n*(n-1)//2)//2
G = nx.gnm_random_graph(n, m, seed=2)
# Upper bound for chromatic number.
c = max([G.degree(i) for i in range(n)]) + 1
# Lower bound for chromatic number.
max_clique = max_clique_approx(G)
print max_clique
print c, ">= X(G) >= ", len(max_clique)



# True value
Z = Bool(n,c)

cost = sum([norm(Z[:,j],'inf') for j in range(c)])
constraints = [Z[i,:] + Z[j,:] <= 1 for i,j in G.edges()]
constraints += [Z*np.ones((c,1)) == 1]
prob = Problem(Minimize(cost), constraints)

prob.solve(solver=GUROBI, verbose=True)
print prob.value

import matplotlib.pyplot as plt
node_color = []
for i in range(n):
    idx = np.argmax(Z[i,:].value)
    node_color.append(idx)
# Repack node colors.
sorted_vals = list(set(sorted(node_color)))
packed_colors = []
for i in range(n):
    idx = sorted_vals.index(node_color[i])
    packed_colors += [colors.hex2color(kelly_colors_hex[idx])]

nx.draw(G, node_color=packed_colors)
plt.show()


# Fix entries in clique.
Z_rows = []
clique_counter = 0
# for i in range(n):
#     if i in max_clique:
#         const = np.zeros((1,c))
#         const[0, clique_counter] = 1
#         Z_rows += [const]
#         clique_counter += 1
#     else:
#         Z_rows += [Choose(1,c,1)]
# Z = vstack(*Z_rows)

a = np.random.randn(n)
a /= np.linalg.norm(a)
# a = -np.ones(n)
Z = Partition(n, c)

# cost = sum([(j+1)*norm(Z[:,j],'inf') for j in range(c)])
cost = sum([norm(Z[:,j],'inf') for j in range(c)])

# Make big sparse matrix for constraints.
import scipy.sparse as sp
V = []
I = []
J = []
count = 0
for i,j in G.edges():
    for k in range(c):
        V += [1.0, 1.0]
        J += [i + k*n, j + k*n]
        I += [count, count]
        count += 1
mat = sp.coo_matrix((V, (I, J)), shape=(count, n*c))
constraints = [mat*vec(Z) <= 1]
# constraints = [Z[i,:] + Z[j,:] <= 1 for i,j in G.edges()]

cost += sum_entries(neg(diff((a.T*Z).T)))
# constraints += [diff((a.T*Z).T) >= 0]
# for i in range(c-1):
#     constraints += [a.T*Z[:,i] <= a.T*Z[:,i+1]]

prob = Problem(Minimize(cost), constraints)

RESTARTS = 8
MAX_ITER = 20
# print prob.solve(method="relax_and_round")

prob.solve(method="NC-ADMM", max_iter=MAX_ITER, random=True, seed=1,
           rho=np.random.uniform(0, 1, size=RESTARTS), prox_polished=False,
           restarts=RESTARTS, sigma=1, polish_depth=100,
           show_progress=True, parallel=True)
print sum([norm(Z[:,j],'inf') for j in range(c)]).value

import matplotlib.pyplot as plt
node_color = []
for i in range(n):
    idx = np.argmax(Z[i,:].value)
    node_color.append(idx)
# Repack node colors.
sorted_vals = list(set(sorted(node_color)))
packed_colors = []
for i in range(n):
    idx = sorted_vals.index(node_color[i])
    packed_colors += [colors.hex2color(kelly_colors_hex[idx])]

nx.draw(G, node_color=packed_colors)
plt.show()
