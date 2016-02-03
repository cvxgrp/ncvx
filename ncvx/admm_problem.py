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
try:
    from Queue import PriorityQueue
except:
    from queue import PriorityQueue

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
    random_z, polish_best, seed, sigma, show_progress,
    prox_polished, polish_depth, lower_bound, alpha, args, kwargs) = data
    noncvx_vars = get_noncvx_vars(orig_prob)

    np.random.seed(idx + seed)
    random.seed(idx + seed)
    # Augmented objective.
    # gamma = cvx.Parameter(sign="positive")
    merit_func = orig_prob.objective.args[0]
    for constr in orig_prob.constraints:
        merit_func += gamma_merit*get_constr_error(constr)
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

    # x^k prev.
    old_vars = {var.id:np.zeros(var.size) for var in orig_prob.variables()}

    best_so_far = [np.inf, {v.id:np.zeros(v.size) for v in orig_prob.variables()}]
    cur_merit = best_so_far[0]
    # ADMM loop
    for k in range(max_iter):
        prev_merit = cur_merit
        try:
            prob.solve(*args, **kwargs)
            # print "post solve cost", idx, k, orig_prob.objective.value
        except cvx.SolverError, e:
            pass
        if prob.status in [cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE]:

            for var in noncvx_vars:
                var.z.value = var.project(alpha*var.value + (1-alpha)*old_vars[var.id] + var.u.value)
                # var.z.value = var.project(np.random.randn(*var.size))
                # var.z.value = var.project(np.random.uniform(0, 1, size=var.size))
                var.u.value += alpha*var.value + (1-alpha)*old_vars[var.id] - var.z.value
            # Update previous iterate.
            old_vars = {var.id:var.value for var in orig_prob.variables()}

            if only_discrete(orig_prob):
                cur_merit, sltn = neighbor_search(merit_func, old_vars, best_so_far, idx, polish_depth)
            else:
                # Try to polish.
                try:
                    polish_opt_val, status = polish(orig_prob, polish_depth, *args, **kwargs)
                    # print "post polish cost", idx, k, orig_prob.objective.value
                except cvx.SolverError, e:
                    polish_opt_val = None
                    status = cvx.SOLVER_ERROR

                # print "polish_opt_val", polish_opt_val
                if status not in [cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE]:
                    # Undo change in var.value.
                    for var in orig_prob.variables():
                        if isinstance(var, NonCvxVariable):
                            var.value = var.z.value
                        else:
                            var.value = old_vars[var.id]

                cur_merit = merit_func.value
                sltn = {v.id:v.value for v in orig_prob.variables()}

            if show_progress and idx == 0:
                print "objective", idx, k, cur_merit, best_so_far[0]
            if cur_merit < best_so_far[0]:
                best_so_far[0] = cur_merit
                best_so_far[1] = sltn

            # # Restore variable values.
            # for var in noncvx_vars:
            #     var.value = var.z.value

            # Termination conditions.
            if best_so_far[0] - lower_bound <= 1e-4:# or \
               # abs(cur_merit - prev_merit) + abs(cur_merit - prev_merit)/(prev_merit + 1) < 1e-4:
                return best_so_far

        else:
            print prob.status
            break

    return best_so_far

def neighbor_search(merit_func, old_vars, global_best, idx, max_depth, show_progress=False):
    nonconvex_vars = get_noncvx_vars(merit_func)
    for var in nonconvex_vars:
        var.value = var.z.value
    best_so_far = [merit_func.value, {v.id:v.value for v in merit_func.variables()}]

    eval_queue = PriorityQueue()
    add_neighbors(eval_queue, merit_func, best_so_far[1], old_vars, 1)
    count = 0
    while not eval_queue.empty():
        merit, _, node_depth, _, sltn = eval_queue.get()
        count += 1
        if merit < best_so_far[0]:# and merit < global_best[0]:
            if idx == 0 and show_progress:
                print merit, count
            best_so_far[0] = merit
            best_so_far[1] = sltn
            if node_depth < max_depth:
                add_neighbors(eval_queue, merit_func, sltn, old_vars, node_depth+1)

    return best_so_far

def add_neighbors(eval_queue, merit_func, sltn, old_vars, depth, max_breadth=100):
    for var in merit_func.variables():
        var.value = sltn[var.id]

    new_nodes = []
    merit_func.value
    for var in get_noncvx_vars(merit_func):
        proj_val = old_vars[var.id] + var.u.value
        for neighbor in var.neighbors(sltn[var.id]):
            var.value = neighbor
            new_nodes.append((merit_func.value,
                             np.linalg.norm(proj_val - neighbor),
                             depth,
                             np.random.randn(),
                             {v.id:v.value for v in merit_func.variables()}))
            var.value = sltn[var.id]

    new_nodes = sorted(new_nodes)
    for i in range(min(max_breadth, len(new_nodes))):
        eval_queue.put(new_nodes[i])

def only_discrete(prob):
    """Does the problem only contain variables in discrete sets?
    """
    for var in prob.variables():
        if not isinstance(var, NonCvxVariable) or not hasattr(var, "_neighbors"):
            return False

    return True

# Use ADMM to attempt non-convex problem.
def admm(self, rho=None, max_iter=50, restarts=5, alpha=1.8,
         random=False, sigma=1.0, gamma=1e6, polish_best=True,
         num_procs=None, parallel=True, seed=1, show_progress=False,
         prox_polished=False, polish_depth=5,
         *args, **kwargs):
    # rho is a list of values, one for each restart.
    if rho is None:
        rho = [np.random.uniform() for i in range(restarts)]
    else:
        assert len(rho) == restarts
    # num_procs is the number of processors to launch.
    if num_procs is None:
        num_procs = multiprocessing.cpu_count()

    # Construct the relaxation.
    if type(self.objective) == cvx.Minimize:
        rel_obj = self.objective
    else:
        rel_obj = -self.objective
    rel_constr = self.constraints
    for var in get_noncvx_vars(self):
        rel_constr += var.relax()
    rel_prob = cvx.Problem(rel_obj, rel_constr)

    lower_bound = rel_prob.solve(*args, **kwargs)
    if show_progress:
        print "lower bound =", lower_bound

    # Algorithm.
    if parallel:
        pool = multiprocessing.Pool(num_procs)
        tmp_prob = cvx.Problem(rel_prob.objective, rel_prob.constraints)
        best_per_rho = pool.map(admm_inner_iter,
            [(idx, tmp_prob, rho_val, gamma, max_iter,
              random, polish_best, seed, sigma, show_progress,
              prox_polished, polish_depth, lower_bound, alpha, args, kwargs) for idx, rho_val in enumerate(rho)])
        pool.close()
        pool.join()
    else:
        best_per_rho = map(admm_inner_iter,
            [(idx, rel_prob, rho_val, gamma, max_iter,
              random, polish_best, seed, sigma, show_progress,
              prox_polished, polish_depth, lower_bound, alpha, args, kwargs) for idx, rho_val in enumerate(rho)])
    # Merge best so far.
    argmin = min([(val[0], idx) for idx, val in enumerate(best_per_rho)])[1]
    best_so_far = best_per_rho[argmin]
    #print "best found", best_so_far[0]
    # Unpack result.
    for var in self.variables():
        var.value = best_so_far[1][var.id]

    residual = cvx.Constant(0)
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

def relax_round_polish(self, gamma=1e4, samples=10, sigma=1, polish_depth=5, seed = 1, *args, **kwargs):
    """Solve the relaxation, then project and polish.
    """
    np.random.seed(seed)
    random.seed(seed)
    # Augment problem.
    residual = cvx.Constant(0)
    for constr in self.constraints:
        residual += get_constr_error(constr)
    merit_func = self.objective.args[0] + gamma*residual
    merit_prob = cvx.Problem(cvx.Minimize(merit_func))
    # Construct and solve the relaxation.
    if type(self.objective) == cvx.Minimize:
        rel_obj = self.objective
    else:
        rel_obj = -self.objective
    rel_constr = self.constraints
    for var in get_noncvx_vars(self):
        rel_constr += var.relax()
    rel_prob = cvx.Problem(rel_obj, rel_constr)
    rel_prob.solve(*args, **kwargs)
    # Save variable values.
    relaxed_values = {v.id:v.value for v in rel_prob.variables()}
    # Randomized projections.
    best_so_far = [np.inf, {}]
    for k in range(samples):
        for var in get_noncvx_vars(rel_prob):
            var_value = relaxed_values[var.id]
            if k == 0:
                var.z.value = var.project(var_value)
            else:
                w = np.random.normal(0, sigma, size=var.size)
                var.z.value = var.project(var_value + w)

        old_vars = {var.id:var.value for var in rel_prob.variables()}
        if only_discrete(rel_prob):
            cur_merit, sltn = neighbor_search(merit_func, old_vars, best_so_far,
                                              0, polish_depth, show_progress)
        else:
            # Try to polish.
            try:
                polish_opt_val, status = polish(rel_prob, polish_depth, *args, **kwargs)
                # print "post polish cost", idx, k, rel_prob.objective.value
            except cvx.SolverError, e:
                polish_opt_val = None
                status = cvx.SOLVER_ERROR

            # print "polish_opt_val", polish_opt_val
            if status not in [cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE]:
                # Undo change in var.value.
                for var in rel_prob.variables():
                    if isinstance(var, NonCvxVariable):
                        var.value = var.z.value
                    else:
                        var.value = old_vars[var.id]

            cur_merit = merit_func.value
            sltn = {v.id:v.value for v in rel_prob.variables()}

        if cur_merit < best_so_far[0]:
            best_so_far[0] = cur_merit
            best_so_far[1] = sltn
    # Unpack result.
    for var in rel_prob.variables():
        var.value = best_so_far[1][var.id]

    return self.objective.value, residual.value

def get_noncvx_vars(prob):
    return sorted([var for var in prob.variables() if getattr(var, "noncvx", False)], key=lambda v: v.id)

def polish(orig_prob, polish_depth=5, *args, **kwargs):
    # Fix noncvx variables and solve.
    for var in get_noncvx_vars(orig_prob):
        var.value = var.z.value
    old_val = None
    for t in range(polish_depth):
        fix_constr = []
        for var in get_noncvx_vars(orig_prob):
            fix_constr += var.restrict(var.value)
        polish_prob = cvx.Problem(orig_prob.objective, orig_prob.constraints + fix_constr)
        polish_prob.solve(*args, **kwargs)
        if polish_prob.status in [cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE] and \
        (old_val is None or (old_val - polish_prob.value)/(old_val + 1) > 1e-3):
            old_val = polish_prob.value
        else:
            break

    return polish_prob.value, polish_prob.status

# Add admm method to cvx Problem.
cvx.Problem.register_solve("NC-ADMM", admm)
cvx.Problem.register_solve("relax-round-polish", relax_round_polish)
