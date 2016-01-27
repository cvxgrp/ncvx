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

n = 10
m = (n*(n-1)//2)//5
G1 = nx.gnm_random_graph(n, m)
G2 = nx.gnm_random_graph(n, m)

S = np.zeros((n*n, n*n))
for i in G1.nodes():
    for j in G1.nodes():
        row_idx = i + j*n
        for k in G2.nodes():
            for l in G2.nodes():
                col_idx = k + l*n
                if (i,j) in G1.edges() and (k,l) in G2.edges():
                    S[row_idx, col_idx] = 1

W = np.random.uniform(0, 1, size=(n,n))

# Solve MIP
X = Bool(n, n)

row_X = vec(X)*np.ones((1,n*n))
col_X = np.ones((n*n,1))*vec(X).T
max_X = min_elemwise(row_X,col_X)
Y = mul_elemwise(S, max_X)
cons = [sum_entries(X, axis=0) <= 1,
        sum_entries(X, axis=1) <= 1]

cost = sum_entries(Y) + trace(W.T*X)
prob = Problem(Maximize(cost), cons)
val = prob.solve(solver=GUROBI, verbose=True)
print "**** true value = ", val

# ADMM heuristic.
X = Assign(n, n)

row_X = vec(X)*np.ones((1,n*n))
col_X = np.ones((n*n,1))*vec(X).T
max_X = min_elemwise(row_X,col_X)
Y = mul_elemwise(S, max_X)
# cons = [max_entries(Y, axis=1) <= vec(X),
#         max_entries(Y, axis=0).T <= vec(X)]

cost = sum_entries(Y) + trace(W.T*X)

prob = Problem(Minimize(-cost))
RESTARTS = 8
MAX_ITER = 50
# print prob.solve(method="relax_and_round")

val, resid = prob.solve(method="admm", max_iter=MAX_ITER, random=True, seed=1,
           rho=np.random.uniform(0, 2, size=RESTARTS),
           restarts=RESTARTS, polish_best=False, sigma=1, nu=0.25,
           show_progress=True, parallel=True, num_proj=1)
print "**** ADMM sltn = ", cost.value
print "ADMM residual = ", resid


