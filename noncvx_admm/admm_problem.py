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

from __future__ import division
from noncvx_variable import NonCvxVariable
from boolean import Boolean
import multiprocessing
import cvxpy as cvx
import numpy as np
import random

# Use ADMM to attempt non-convex problem.
def admm_basic(self, rho=0.5, iterations=5, random=False, *args, **kwargs):
    noncvx_vars = []
    for var in self.variables():
        if getattr(var, "noncvx", False):
            noncvx_vars += [var]
            var.init_z(random=False)
    # Form ADMM problem.
    obj = self.objective.args[0]
    for var in noncvx_vars:
        obj += (rho/2)*cvx.sum_entries(cvx.square(var - var.z + var.u))
    prob = cvx.Problem(cvx.Minimize(obj), self.constraints)
    # ADMM loop
    for i in range(iterations):
        result = prob.solve(*args, **kwargs)
        print "relaxation", result
        for idx, var in enumerate(noncvx_vars):
            var.z.value = var.project(var.value + var.u.value)
            # print idx, var.z.value, var.value, var.u.value
            var.u.value += var.value - var.z.value
    return polish(self, *args, **kwargs)

def get_constr_error(constr):
    if isinstance(constr, cvx.constraints.EqConstraint):
        error = cvx.abs(constr.args[0] - constr.args[1])
    elif isinstance(constr, cvx.constraints.LeqConstraint):
        error = cvx.pos(constr.args[0] - constr.args[1])
    elif isinstance(constr, cvx.constraints.PSDConstraint):
        mat = constr.args[0] - constr.args[1]
        error = cvx.neg(cvx.lambda_min(mat + mat.T)/2)
    return cvx.sum_entries(error)

def admm_inner_iter(data):
    (idx, orig_prob, rho_val, gamma_merit, max_iter,
    random_z, polish_best, seed, sigma, show_progress, args, kwargs) = data
    noncvx_vars = get_noncvx_vars(orig_prob)

    np.random.seed(idx + seed)
    random.seed(idx + seed)
    # Augmented objective.
    gamma = cvx.Parameter(sign="positive")
    merit_func = orig_prob.objective.args[0]
    for constr in orig_prob.constraints:
        merit_func += gamma*get_constr_error(constr)
    # Form ADMM problem.
    obj = orig_prob.objective.args[0]
    for var in noncvx_vars:
        obj += (rho_val/2)*cvx.sum_squares(var - var.z + var.u)
    prob = cvx.Problem(cvx.Minimize(obj), orig_prob.constraints)

    for var in noncvx_vars:
        # var.init_z(random=random_z)
        # var.init_u()
        if idx == 0 or not random_z:
            var.z.value = np.zeros(var.size)
        else:
            var.z.value = np.random.normal(0, sigma, var.size)
        var.u.value = np.zeros(var.size)

    best_so_far = [np.inf, {}]
    # ADMM loop
    for k in range(max_iter):
        gamma.value = gamma_merit#min((1.1**k), gamma_merit)
        try:
            prob.solve(*args, **kwargs)
            # print "post solve cost", idx, k, orig_prob.objective.value
        except cvx.SolverError, e:
            pass
        if prob.status in [cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE]:
            # print rho_val, k, best_so_far[0]
            for var in noncvx_vars:
                # if isinstance(var, Boolean):
                #     var.z.value = np.random.uniform(size=var.size) < var.value + var.u.value
                # else:
                var.z.value = var.project(var.value + var.u.value)
                # var.z.value = var.project(var.value + var.u.value + \
                #     np.random.normal(scale=sigma, size=var.size))
                var.u.value += var.value - var.z.value
                # if k == 0:
                #     print outer_iter
                #     print var.value
                #     print var.z.value
            old_vars = {var.id:var.value for var in orig_prob.variables()}
            if polish_best:
                # Try to polish.
                try:
                    polish_opt_val, status = polish(orig_prob, *args, **kwargs)
                    # print "post polish cost", idx, k, orig_prob.objective.value
                except cvx.SolverError, e:
                    polish_opt_val = None
                    status = cvx.SOLVER_ERROR
            # print "polish_opt_val", polish_opt_val
            if not polish_best or status not in [cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE]:
                # Undo change in var.value.
                for var in orig_prob.variables():
                    if isinstance(var, NonCvxVariable):
                        var.value = var.z.value
                    else:
                        var.value = old_vars[var.id]

            for var in orig_prob.variables():
                if isinstance(var, NonCvxVariable):
                    var.z.value = var.value

            gamma.value = gamma_merit
            merit = merit_func.value
            # for constr in orig_prob.constraints:
            #     if cvx.sum_entries(constr.violation).value > 1e-1:
            #         merit += gamma*cvx.sum_entries(constr.violation).value
            if show_progress:
                print "objective", idx, k, merit
            if merit <= best_so_far[0]:
                best_so_far[0] = merit
                best_so_far[1] = {v.id:v.value for v in prob.variables()}

            # # Restore variable values.
            # for var in noncvx_vars:
            #     var.value = var.z.value

        else:
            print prob.status
            break

    return best_so_far

# Use ADMM to attempt non-convex problem.
def admm(self, rho=None, max_iter=50, restarts=5,
         random=False, sigma=1.0, gamma=1e6, polish_best=True,
         num_procs=None, parallel=True, seed=1, show_progress=False,
         *args, **kwargs):
    # rho is a list of values, one for each restart.
    if rho is None:
        rho = [np.random.uniform() for i in range(restarts)]
    else:
        assert len(rho) == restarts
    # num_procs is the number of processors to launch.
    if num_procs is None:
        num_procs = multiprocessing.cpu_count()

    # Solve the relaxation.
    rel_val = self.solve(*args, **kwargs)
    print "lower bound", rel_val

    # Algorithm.
    if parallel:
        pool = multiprocessing.Pool(num_procs)
        tmp_prob = cvx.Problem(self.objective, self.constraints)
        best_per_rho = pool.map(admm_inner_iter,
            [(idx, tmp_prob, rho_val, gamma, max_iter,
              random, polish_best, seed, sigma, show_progress, args, kwargs) for idx, rho_val in enumerate(rho)])
        pool.close()
        pool.join()
    else:
        best_per_rho = map(admm_inner_iter,
            [(idx, self, rho_val, gamma, max_iter,
              random, polish_best, seed, sigma, show_progress, args, kwargs) for idx, rho_val in enumerate(rho)])
    # Merge best so far.
    argmin = min([(val[0], idx) for idx, val in enumerate(best_per_rho)])[1]
    best_so_far = best_per_rho[argmin]
    print "best found", best_so_far[0]
    # Unpack result.
    for var in self.variables():
        var.value = best_so_far[1][var.id]

    residual = 0
    for constr in self.constraints:
        residual += get_constr_error(constr)

    return self.objective.value, residual.value

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

def relax_project_polish(self, gamma=1e4, samples=10, sigma=1, *args, **kwargs):
    """Solve the relaxation, then project and polish.
    """
    # Augment problem.
    residual = 0
    for constr in self.constraints:
        residual += get_constr_error(constr)
    merit_func = self.objective.args[0] + gamma*residual
    merit_prob = cvx.Problem(cvx.Minimize(merit_func))
    # solve relaxation.
    self.solve(*args, **kwargs)
    # Save variable values.
    relaxed_values = {v.id:v.value for v in self.variables()}
    # Randomized projections.
    best_so_far = [np.inf, {}]
    for k in range(samples):
        for var in get_noncvx_vars(self):
            var_value = relaxed_values[var.id]
            if k == 0:
                var.z.value = var.project(var_value)
            else:
                w = np.random.normal(0, sigma, size=var.size)
                var.z.value = var.project(var_value + w)
            obj_value, status = polish(self, *args, **kwargs)
            if status not in [cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE]:
                # Undo change in var.value.
                for var in self.variables():
                    if isinstance(var, NonCvxVariable):
                        var.value = var.z.value
                    else:
                        var.value = relaxed_values[var.id]
            merit = merit_func.value
            if merit < best_so_far[0]:
                best_so_far[0] = merit
                best_so_far[1] = {v.id:v.value for v in self.variables()}
    # Unpack result.
    for var in self.variables():
        var.value = best_so_far[1][var.id]

    return self.objective.value, residual.value

def repeated_rr(self, tau_init=1, tau_max=250, delta=1.1, max_iter=10,
                random=False, abs_eps=1e-4, rel_eps=1e-4, *args, **kwargs):
    noncvx_vars = get_noncvx_vars(self)
    # Form ADMM problem.
    tau_param = cvx.Parameter(sign="positive")
    tau_param.value = 0
    obj = self.objective.args[0]
    for var in noncvx_vars:
        obj = obj + (tau_param)*cvx.norm(var - var.z, 1)
    prob = cvx.Problem(cvx.Minimize(obj), self.constraints)
    # Algorithm.
    best_so_far = [np.inf, np.inf, {}]
    for var in noncvx_vars:
        var.init_z(random=random)
    # ADMM loop
    for k in range(max_iter):
        try:
            prob.solve(*args, **kwargs)
        except cvx.SolverError, e:
            pass
        if prob.status is cvx.OPTIMAL:
            opt_val = self.objective.value
            noncvx_inf = total_dist(noncvx_vars)
            print "iter ", k
            print "tau", tau_param.value
            print "original obj val", opt_val
            print "augmented value", prob.value
            print "noncvx_inf", noncvx_inf

            # Is the infeasibility better than best_so_far?
            error = get_error(noncvx_vars, abs_eps, rel_eps)
            if is_better(noncvx_inf, opt_val, best_so_far, error):
                best_so_far[0] = noncvx_inf
                best_so_far[1] = opt_val
                best_so_far[2] = {v.id:v.value for v in prob.variables()}

            for var in noncvx_vars:
                var.z.value = var.project(var.value)
        else:
            break

        # Update tau.
        tau_param.value = max(tau_init, min(tau_param.value*(delta**k), tau_max))

        # Convergence criteria.
        # TODO

        # # Polish the best iterate.
        # for var in prob.variables():
        #     var.value = best_so_far[2][var.id]
        # opt_val, status = polish(self, *args, **kwargs)
        # if status is cvx.OPTIMAL:
        #    error = get_error(noncvx_vars, eps, rel_eps)
        #    if is_better(0, opt_val, best_so_far, error):
        #         best_so_far[0] = 0
        #         best_so_far[1] = opt_val
        #         best_so_far[2] = {v.id:v.value for v in prob.variables()}

    # Unpack result.
    for var in prob.variables():
        var.value = best_so_far[2][var.id]
    error = get_error(noncvx_vars, abs_eps, rel_eps)
    if best_so_far[0] < error:
        return best_so_far[1]
    else:
        return np.inf

# Use ADMM to attempt non-convex problem.
def admm2(self, rho=0.5, iterations=5, random=False, *args, **kwargs):
    noncvx_vars = []
    for var in self.variables():
        if getattr(var, "noncvx", False):
            var.init_z(random=random)
            noncvx_vars += [var]
    # Form ADMM problem.
    obj = self.objective.args[0]
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

def get_noncvx_vars(prob):
    return [var for var in prob.variables() if getattr(var, "noncvx", False)]


def polish(prob, *args, **kwargs):
    # Fix noncvx variables and solve.
    old_val = None
    while True:
        fix_constr = []
        for var in get_noncvx_vars(prob):
            fix_constr += var.restrict(var.z.value)
        prob = cvx.Problem(prob.objective, prob.constraints + fix_constr)
        prob.solve(*args, **kwargs)
        # print "polishing iterate", prob.value
        if old_val is None or (old_val - prob.value)/old_val > 1e-2:
            old_val = prob.value
        else:
            break

    return prob.value, prob.status

# Add admm method to cvx Problem.
cvx.Problem.register_solve("admm_basic", admm_basic)
cvx.Problem.register_solve("admm", admm)
cvx.Problem.register_solve("admm2", admm2)
cvx.Problem.register_solve("relax_project_polish", relax_project_polish)
cvx.Problem.register_solve("repeated_rr", repeated_rr)
