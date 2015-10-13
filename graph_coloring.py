from __future__ import division
from cvxpy import *
from noncvx_admm import *
import networkx as nx
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

n = 40
m = (n*(n-1)//2)//5
G = nx.gnm_random_graph(n, m, seed=2)
# Upper bound for chromatic number.
c = max([G.degree(i) for i in range(n)]) + 1
# Lower bound for chromatic number.
max_clique = max((len(clique), clique) for clique in nx.find_cliques(G))
print c, ">= X(G) >= ", max_clique[0]
# Fix entries in clique.
# c = n
Z_rows = []
clique_counter = 0
for i in range(n):
    if i in max_clique[1]:
        const = np.zeros((1,c))
        const[0, clique_counter] = 1
        Z_rows += [const]
        clique_counter += 1
    else:
        Z_rows += [Choose(1,c,1)]
Z = vstack(*Z_rows)

cost = sum([(j+1)*norm(Z[:,j],'inf') for j in range(c)])
constraints = [Z[i,:] + Z[j,:] <= 1 for i,j in G.edges()]
prob = Problem(Minimize(cost), constraints)

RESTARTS = 5
MAX_ITER = 50
# print prob.solve(method="relax_and_round")

prob.solve(method="admm", max_iter=MAX_ITER, random=True,
           rho=np.random.uniform(5, 10, size=RESTARTS), seed=1,
           restarts=RESTARTS, polish_best=False)
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
