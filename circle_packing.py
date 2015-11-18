from __future__ import division
import cvxpy as cp, numpy as np, cvxopt, matplotlib.pyplot as plt, pickle, random
import noncvx_admm as ncvx

N = 12 # number of circles
np.random.seed(0)
random.seed(0)

#define variables.
x_vals = [cp.Variable() for i in range(N)]
y_vals = [cp.Variable() for i in range(N)]

# # Initialize values.
# for i in range(N):
#     x_vals[i].value = np.random.uniform()
#     y_vals[i].value = np.random.uniform()

# # Get distances.
# min_dist = np.inf
# for i in range(N):
#     for j in range(i+1,N):
#         diff = cp.vstack(x_vals[i] - x_vals[j],
#                          y_vals[i] - y_vals[j])
#         dist = cp.norm(diff).value
#         min_dist = min(min_dist, dist)
# # Spread out so further apart.
# for i in range(N):
#     x_vals[i].value /= min_dist
#     y_vals[i].value /= min_dist

l = cp.max_elemwise( *(x_vals + y_vals) ) + 0.5
objective = cp.Minimize(l)
constraints = []
for i in xrange(N):
    constraints  += [0.5 <= x_vals[i],
                     0.5 <= y_vals[i]]
soc_bound_vars = []
for i in xrange(N-1):
    for j in xrange(i+1,N):
        t = cp.Variable()
        soc_bound_vars.append(ncvx.Annulus(2, 1, N))
        constraints += [
            cp.vstack(x_vals[i] - x_vals[j],
                      y_vals[i] - y_vals[j]) == soc_bound_vars[-1],
            ]

# # Symmetry breaking constraints
# for i in range(N-1):
#     constraints += [x_vals[i] + y_vals[i] <= x_vals[i+1] + y_vals[i+1]]
# for i in range(1,N-1):
#     constraints += [(np.sqrt(np.pi*i) - 1)/2 <= (x_vals[i+1] + y_vals[i+1])/np.sqrt(2)]
#     constraints += [(x_vals[i+1] + y_vals[i+1])/np.sqrt(2) <= np.sqrt(np.pi*i/2 + 2)]
prob = cp.Problem(objective, constraints)

# Initialize args.

# for var in cvx_prob.variables():
#     var.value = l*np.random.uniform()
# r.value = 1
# for i in range(N):
#     x_vals[i].value = 5*i#p_c[0,i]
#     y_vals[i].value = 5*i#p_c[1,i]

# for iteration in xrange(max_iter):
RESTARTS = 5
ITERS = 10
result = prob.solve(method="admm", max_iter=ITERS, random=True, seed=1,
           rho=np.random.uniform(0, 1, size=RESTARTS),
           num_proj=1, sigma=1,
           restarts=RESTARTS, polish_best=True, prox_polished=False,
           show_progress=True, parallel=True, verbose=False)
print result
print "bounding box dim = ", l.value
print "circle diameter = ", 1/l.value
print "circle radius = ", 1/(2*l.value)
print "percent coverage = ", N*np.pi*0.25/cp.square(l).value
# for var in soc_bound_vars:
#     print var.value
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


# variables for plotting.
pi = np.pi
circ = np.linspace(0,2*pi)
x_border = [0, l.value, l.value, 0, 0]
y_border = [0, 0, l.value, l.value, 0]


#plot the circles
for i in xrange(N):
    plt.plot(x_vals[i].value+0.5*np.cos(circ),y_vals[i].value+0.5*np.sin(circ),'b')
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