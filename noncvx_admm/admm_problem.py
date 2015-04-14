"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

from noncvx_variable import NonCvxVariable
import cvxpy as cvx
import numpy as np

# Use ADMM to attempt non-convex problem.
def admm(self, rho=None, max_iter=5, restarts=1,
         random=False, eps=1e-4, rel_eps=1e-4,
         *args, **kwargs):
    # rho is a list of values, one for each restart.
    if rho is None:
        rho = [np.random.uniform() for i in range(restarts)]
    else:
        assert len(rho) == restarts
    # Setup the problem.
    noncvx_vars = []
    for var in self.variables():
        if isinstance(var, NonCvxVariable):
            noncvx_vars += [var]
    # Form ADMM problem.
    rho_param = cvx.Parameter(sign="positive")
    obj = self.objective._expr
    for var in noncvx_vars:
        obj = obj + (rho_param/2)*cvx.sum_squares(var - var.z + var.u)
    prob = cvx.Problem(cvx.Minimize(obj), self.constraints)
    # Algorithm.
    best_so_far = [np.inf, np.inf, {}]
    for rho_val in rho:
        for var in noncvx_vars:
            var.init_z(random=random)
        # ADMM loop
        for k in range(max_iter):
            rho_param.value = rho_val#*(1.01)**k
            try:
                prob.solve(*args, **kwargs)
            except cvx.SolverError, e:
                pass
            if prob.status is cvx.OPTIMAL:
                opt_val = self.objective.value
                noncvx_inf = total_dist(noncvx_vars)

                # Is the infeasibility better than best_so_far?
                error = get_error(noncvx_vars, eps, rel_eps)
                if is_better(noncvx_inf, opt_val, best_so_far, error):
                    best_so_far[0] = noncvx_inf
                    best_so_far[1] = opt_val
                    best_so_far[2] = {v.id:v.value for v in prob.variables()}
                for var in noncvx_vars:
                    var.z.value = var.project(var.value + var.u.value)
                    var.u.value += var.value - var.z.value
                    var.value = var.z.value
            else:
                break

            # Convergence criteria.
            # TODO

        # Polish the best iterate.
        for var in prob.variables():
            var.value = best_so_far[2][var.id]
        opt_val, status = polish(self, *args, **kwargs)
        if status is cvx.OPTIMAL:
           error = get_error(noncvx_vars, eps, rel_eps)
           if is_better(0, opt_val, best_so_far, error):
                best_so_far[0] = 0
                best_so_far[1] = opt_val
                best_so_far[2] = {v.id:v.value for v in prob.variables()}

    # Unpack result.
    for var in prob.variables():
        var.value = best_so_far[2][var.id]
    error = get_error(noncvx_vars, eps, rel_eps)
    if best_so_far[0] < error:
        return best_so_far[1]
    else:
        return np.inf

def total_dist(noncvx_vars):
    """Get the total distance from the noncvx_var values
    to the nonconvex sets.
    """
    total = 0
    for var in noncvx_vars:
        total += var.dist(var.value)
    return total

def get_error(noncvx_vars, eps, rel_eps):
    """The error bound for comparing infeasibility.
    """
    error = sum([cvx.norm(cvx.vec(var)) for var in noncvx_vars]).value
    return eps + rel_eps*error

def is_better(noncvx_inf, opt_val, best_so_far, error):
    """Is the current result better than best_so_far?
    """
    inf_diff = best_so_far[0] - noncvx_inf
    return (inf_diff > error) or \
           (abs(inf_diff) <= error and opt_val < best_so_far[1])


# Use ADMM to attempt non-convex problem.
def admm2(self, rho=0.5, iterations=5, random=False, *args, **kwargs):
    noncvx_vars = []
    for var in self.variables():
        if getattr(var, "noncvx", False):
            var.init_z(random=random)
            noncvx_vars += [var]
    # Form ADMM problem.
    obj = self.objective._expr
    for var in noncvx_vars:
        obj = obj + (rho/2)*cvx.sum_squares(var - var.z + var.u)
    prob = cvx.Problem(cvx.Minimize(obj), self.constraints)
    # ADMM loop
    best_so_far = np.inf
    for i in range(iterations):
        result = prob.solve(*args, **kwargs)
        for var in noncvx_vars:
            var.z.value = var.project(var.value + var.u.value)
            var.u.value += var.value - var.z.value
        polished_opt = polish(self, noncvx_vars, *args, **kwargs)
        if polished_opt < best_so_far:
            best_so_far = polished_opt
            print best_so_far
    return best_so_far

def polish(prob, *args, **kwargs):
    # Fix noncvx variables and solve.
    fix_constr = []
    for var in prob.variables():
        if getattr(var, "noncvx", False):
            fix_constr += var.fix(var.z.value)
    prob = cvx.Problem(prob.objective, prob.constraints + fix_constr)
    prob.solve(*args, **kwargs)
    return prob.value, prob.status

# Add admm method to cvx Problem.
cvx.Problem.register_solve("admm", admm)
cvx.Problem.register_solve("admm2", admm2)
