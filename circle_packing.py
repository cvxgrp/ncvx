from __future__ import division
from cvxpy import *
from noncvx_admm import *
import numpy as np

N = 4
# Differences
Y_list = [ExtrBall(2) for i in range(N-1)]
Y = hstack(*Y_list)
# Positions.
X_list = [np.array([0,0])]
for i in range(1,N):
    X_list += [Y[:,i-1] + X_list[-1]]
X = hstack(*X_list)

min_x0 = min_entries(X[0,:]) - 0.5
max_x0 = max_entries(X[0,:]) + 0.5
l_x0 = max_x0 - min_x0
min_x1 = min_entries(X[1,:]) - 0.5
max_x1 = max_entries(X[1,:]) + 0.5
l_x1 = max_x1 - min_x1
l = max_elemwise(l_x0, l_x1)

prob = Problem(Minimize(l))

RESTARTS = 2
ITERS = 1
result = prob.solve(method="admm", max_iter=ITERS,
                    restarts=RESTARTS, random=True)
print result

x_border = [min_x0.value, max_x0.value, max_x0.value,
            min_x0.value, min_x0.value]
y_border = [min_x1.value, min_x1.value, max_x1.value,
            max_x1.value, min_x1.value]

#plot the circles
circ = np.linspace(0,2*np.pi)
import matplotlib.pyplot as plt
for i in xrange(N):
    plt.plot(X[0,i].value+0.5*np.cos(circ),X[1,i].value+0.5*np.sin(circ),'b')
    # if(colors[i] == 0):
    #     plt.plot(p.value[0,i]+r.value*np.cos(circ),p.value[1,i]+r.value*np.sin(circ),'b')
    # else:
    #     plt.plot(p.value[0,i]+r.value*np.cos(circ),p.value[1,i]+r.value*np.sin(circ),'r')
plt.plot(x_border,y_border,'g');
plt.axes().set_aspect('equal')
    # title = 'Iteration ' + repr(iteration) +' Tau is '+str.format('{0:.3f}',Tau)+'\n objective value is '+str.format('{0:.2f}',r.value*r.value*pi*N)+'\n violation is '+str.format('{0:.2f}',sum(slacks).value)
    # plt.title(title)
    # display.clear_output()
plt.show();