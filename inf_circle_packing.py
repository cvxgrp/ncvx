import cvxpy as cp, numpy as np, cvxopt, matplotlib.pyplot as plt, pickle
import noncvx_admm as ncvx

N = 2 # number of circles
l = 10 # side length of square

#algorithm parameters
delta = 1e-3 # end condition tolerance
Tau_inc = 1.5 # amount to increase tau
Tau_init = 0.5 # initial tau value
Tau_max = 1e6 # maximum tau value
max_iter = 10 # value to limit the maximum number of iterations attempted
np.random.seed(0)

#define variables.
r = cp.Variable(1);
x_vals = [cp.Variable() for i in range(N)]
y_vals = [cp.Variable() for i in range(N)]

# variables for plotting.
pi = np.pi
circ = np.linspace(0,2*pi)
x_border = [0, l, l, 0, 0]
y_border = [0, 0, l, l, 0]

prev_val = np.infty;
Tau = Tau_init
r_c = np.random.uniform(1)
p_c = l*np.random.uniform(size=(2,N))

objective = cp.Minimize(cp.max_elemwise( *(x_vals + y_vals) ))
constraints = []
for i in xrange(N):
    constraints  += [0 <= x_vals[i],
                     0 <= y_vals[i]]
soc_bound_vars = []
for i in xrange(N-1):
    for j in xrange(i+1,N):
        t = cp.Variable()
        soc_bound_vars.append(ncvx.ExtrBall(2))
        constraints += [
            cp.vstack(x_vals[i] - x_vals[j],
                      y_vals[i] - y_vals[j]) == soc_bound_vars[-1],
            ]
prob = cp.Problem(objective, constraints)

# Initialize args.

# for var in cvx_prob.variables():
#     var.value = l*np.random.uniform()
# r.value = 1
# for i in range(N):
#     x_vals[i].value = 5*i#p_c[0,i]
#     y_vals[i].value = 5*i#p_c[1,i]

# for iteration in xrange(max_iter):
result = prob.solve(method="consensus", max_iters=100,
                    restarts=1, random=True, rho=[1])
print result
for var in soc_bound_vars:
    print var.value
# #solver error detected
# if(result == 'solver_error'):
#     break

# #Plotting Code
# #determine if constraints are violated, and if they are indicate that a circle should be red
# colors = [0]*N;
# for i in xrange(N-1):
#     if(slack_new.value[0,i] > delta or slack_new.value[0,N+i] > delta or slack_new.value[1,i] > delta or slack_new.value[1,N+i] > delta):
#         colors[i] = 1
#     for j in xrange(i+1,N):
#         if(slack.value[((N-1)+(N-i))*i/2+j-i-1] > 1e-3):
#             colors[i] = 1
#             colors[j] = 1

#plot the circles
for i in xrange(N):
    plt.plot(x_vals[i].value+r.value*np.cos(circ),y_vals[i].value+r.value*np.sin(circ),'b')
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
    # # p_c = p.value
    # # r_c = r.value
    # Tau = min(Tau*Tau_inc,Tau_max)
    # if (np.abs(prev_val-result) <= delta and (sum(slacks).value < delta or Tau == Tau_max)):
    #         break
    # prev_val = result