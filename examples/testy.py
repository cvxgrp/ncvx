from cvxpy import *

from noncvx_admm import *

import numpy

n=10

k=2

p=2

#generate data

numpy.random.seed(1)

a0 = numpy.random.randn(n,k)

b0 = numpy.random.randn(k,n)

L0 = numpy.dot(a0,b0)

E0 = numpy.zeros((n,n))

E0[3,5]=1

E0[7,9]=1

X=L0+E0

#solve

L = Rank(n,n,k)

E = reshape(Card(n*n,p,p), n,n)

X=L0+E0

cost=sum_squares(X-L-E)

prob=Problem(Minimize(cost))

prob.solve(method="admm",max_iter=50, restarts=5, random=True)