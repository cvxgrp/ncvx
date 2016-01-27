from __future__ import division
from cvxpy import *
from ncvx import *
import random
import numpy as np

random.seed(1)
np.random.seed(1)
# 3-SAT problem solved with non-convex ADMM

# num_vars_list = np.arange(10,110,10)
# num_clauses_list = np.arange(10,430,10)
num_vars_list = np.arange(10,110,10)
num_clauses_list = np.arange(10,440,10)

data = np.zeros((len(num_clauses_list), len(num_vars_list)))
TRIALS = 10

for idx_1, num_vars in enumerate(num_vars_list):
    for idx_2, num_clauses in enumerate(num_clauses_list):
        if num_clauses/num_vars < 4.25:
            print num_vars, num_clauses
            successes = 0
            for k in range(TRIALS):
                # Randomly generate a feasible 3-SAT problem.
                while True:
                    # The 3-SAT clauses.
                    A = np.zeros((num_clauses,num_vars))
                    b = np.zeros((num_clauses, 1))
                    for i in range(num_clauses):
                        clause_vars = random.sample(range(num_vars), 3)
                        # Which variables are negated in the clause?
                        negated = np.array([random.random() < 0.5 for j in range(3)])
                        A[i, clause_vars] = 2*negated - 1
                        b[i] = sum(negated) - 1
                    # print "Generated %d clauses." % CLAUSES

                    x = Bool(num_vars)
                    # cost = sum([abs(v-0.5) for v in vars])
                    prob = Problem(Minimize(0), [A*x <= b])
                    prob.solve(solver=GUROBI)
                    if prob.status != INFEASIBLE:
                        break
                    else:
                        print "INFEASIBLE"

                # WEIRD. only works if rho varies.
                x = Boolean(num_vars)
                prob = Problem(Minimize(0), [A*x <= b])
                RESTARTS = 10
                result = prob.solve(method="admm", restarts=RESTARTS,
                                    # rho=np.random.uniform(size=RESTARTS),
                                    rho=RESTARTS*[10],
                                    max_iter=100, solver=ECOS, random=True,
                                    polish_best=False)

                satisfied = (A*x.value <= b).sum()
                percent_satisfied = 100*satisfied/num_clauses
                if percent_satisfied == 100:
                    successes += 1
                # print "%s%% of the clauses were satisfied." % percent_satisfied


                # print prob.solve(method="relax_and_round")
                # satisfied = (A*x.value <= b).sum()
                # percent_satisfied = 100*satisfied/CLAUSES
                # print "%s%% of the clauses were satisfied." % percent_satisfied
            print successes/TRIALS
            data[idx_2, idx_1] = successes/TRIALS

import matplotlib.pyplot as plt
# Plot results.
fig, ax = plt.subplots()
heatmap = ax.pcolor(data, cmap=plt.cm.Blues)
fig.colorbar(heatmap)

# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
ax.set_yticks(np.arange(data.shape[0])[4::5]+0.5, minor=False)

# want a more natural, table-like display
# ax.invert_yaxis()
# ax.xaxis.tick_top()

ax.set_xticklabels(num_vars_list, minor=False)
ax.set_yticklabels(num_clauses_list[4::5], minor=False)
plt.savefig("three_sat.png")
plt.show()
