from cvxpy import *
from noncvx_admm import *
import numpy
import matplotlib.pyplot as plt

n=10 #matrix size
k=3  #rank of L0
p=[1,3,5,7,9]  #cardinality of E0
times=1  #times of random trials
numpy.random.seed(1)
error_L=numpy.zeros((times,len(p)))
error_E=numpy.zeros((times,len(p)))
cost_val=numpy.zeros((times,len(p)))
for i in range(times):
                #generate L0
		a0 = numpy.random.randn(n,k)
		b0 = numpy.random.randn(k,n)
		L0 = numpy.dot(a0,b0)
		for p_ind in range(len(p)):
                        #generate E0
                        E0 = numpy.zeros((n*n,1))
                        arr = numpy.arange(100)
                        numpy.random.shuffle(arr)
                        pos = arr[0:p[p_ind]]
                        E0[pos] = numpy.random.randn(p[p_ind],1)
                        # pos = arr[0:10]
                        # E0[pos] = numpy.random.randn(10,1)
                        E0 = reshape(E0,n,n)
                        #generate X
                        X=L0+E0
                        #solve
                        L = Rank(n,n,k)
                        E = reshape(Card(n*n,p[p_ind],1000), n,n)
                        cost=sum_squares(X-L-E)
                        prob=Problem(Minimize(cost))
                        print prob.solve(method="relax_and_round")
                        prob.solve(method="admm",max_iter=100, restarts=5, random=True)

                        print numpy.linalg.svd(L.value.A, compute_uv=0)
                        error_L[i,p_ind] = sum_squares(L0-L).value/sum_squares(L0).value
                        print("error_L=",error_L[i,p_ind])
                        error_E[i,p_ind]=sum_squares(E0-E).value/sum_squares(E0).value
                        print("error_E=",error_E[i,p_ind])
                        cost_val[i,p_ind]= sum_squares(X-L-E).value
                        print("cost=",prob.value)
plt.figure(figsize=(6,6))
plt.plot(p,sum(error_L)/times)
plt.xlabel('cardinality', fontsize=16)
plt.ylabel('error_L', fontsize=16)
plt.show()

plt.figure(figsize=(7,7))
plt.plot(p,sum(error_E)/times)
plt.xlabel('cardinality', fontsize=16)
plt.ylabel('error_E', fontsize=16)
plt.show()

plt.figure(figsize=(8,8))
plt.plot(p,sum(cost_val)/times)
plt.xlabel('cardinality', fontsize=16)
plt.ylabel('cost', fontsize=16)
plt.show()
