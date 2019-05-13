from cvxpy import *
from ncvx import *
import random
import numpy as np

random.seed(1)
np.random.seed(1)
# 3-SAT problem solved with non-convex ADMM

# num_vars_list = np.arange(10,110,10)
# num_clauses_list = np.arange(10,430,10)
#num_vars_list = np.arange(10,110,10) #10,110,10
#num_clauses_list = np.arange(10,440,10) #10,440,10

#data = np.zeros((len(num_clauses_list), len(num_vars_list)))
#TRIALS = 10

# Generating problem data
num_clauses = 150
num_vars = 60;

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
    ## Verifying the problem is feasible. (Uncomment to use Gurobi.)
    x = Bool(num_vars)
    prob = Problem(Minimize(0), [A*x <= b])
    # prob.solve(solver=GUROBI)
    if prob.status != INFEASIBLE:
        break
    else:
        print("INFEASIBLE")

# NC-ADMM Heuristic
x = Boolean(num_vars)
prob = Problem(Minimize(0), [A*x <= b])
RESTARTS = 10
result = prob.solve(method="NC-ADMM")

satisfied = (A*x.value <= b).sum()
percent_satisfied = 100*satisfied/num_clauses
print("%s%% of the clauses were satisfied." % percent_satisfied)
