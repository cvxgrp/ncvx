from cvxpy import *
from ncvx import *
# from ncvx.boolean import Boolean
# import ncvx.branch_and_bound
import cvxopt
import numpy as np
np.random.seed(1)

# x = Boolean(3, name='x')
# A = cvxopt.matrix([1,2,3,4,5,6,7,8,9], (3, 3), tc='d')
# z = cvxopt.matrix([3, 7, 9])

n = 200
m = 400
x = Boolean(n)
A = np.random.randn(m, n)
true_x = np.random.uniform(size=n) > 0.5
b = A.dot(true_x)

prob = Problem(Minimize(sum_squares(A*x - b)))
prob.solve(method="relax_and_round")
# prob.solve(method="admm", max_iter=50, restarts=5)

print x.value.sum()
print (x-true_x).value.sum()
print prob.value

# even a simple problem like this introduces too many variables
# y = Boolean()
# Problem(Minimize(square(y - 0.5))).branch_and_bound()
