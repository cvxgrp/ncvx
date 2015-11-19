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

from .boolean import Boolean
import cvxopt
import numpy as np
from itertools import product
from cvxpy import diff, sum_entries, Variable
import cvxpy.lin_ops.lin_utils as lu

class Partition(Boolean):
    """ A boolean matrix with exactly one 1 in each row.
    """
    def __init__(self, rows, cols, *args, **kwargs):
        super(Partition, self).__init__(rows, cols, *args, **kwargs)

    def _project(self, matrix):
        """The largest value in each row is set to 1.
        """
        result = np.zeros(self.size)
        for i in range(self.size[0]):
            idx = np.argmax(matrix[i,:])
            result[i, idx] = 1
        return result
        # ordering = self.a.T.dot(result)
        # indices = np.argsort(ordering)
        # return result[:,indices]
        # import cvxpy as cvx
        # X = cvx.Bool(*self.size)
        # constr = [cvx.sum_entries(X, axis=1) == 1, cvx.diff((self.a.T*X).T) >= 0]
        # prob = cvx.Problem(cvx.Maximize(cvx.trace(matrix.T*X)), constr)
        # prob.solve(solver=cvx.GUROBI, timeLimit=10)
        # return X.value

    # def _neighbors(self, matrix):
    #     neighbors_list = []
    #     idxs = np.argmax(matrix, axis=1)
    #     for i in range(self.size[0]):
    #         for j in range(self.size[1]):
    #             if j != idxs[i]:
    #                 new_mat = matrix.copy()
    #                 new_mat[i,j] = 1
    #                 new_mat[i,idxs[i]] = 0
    #                 neighbors_list += [new_mat]
    #     return neighbors_list

    def _neighbors(self, matrix):
        neighbors_list = []
        idxs = np.argmax(matrix, axis=1)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if j != idxs[i] and abs(j - idxs[i]) <= 1:
                    new_mat = matrix.copy()
                    new_mat[i,j] = 1
                    new_mat[i,idxs[i]] = 0
                    neighbors_list += [new_mat]
        return neighbors_list

    # def _neighbors(self, matrix):
    #     neighbors_list = []
    #     for i in range(25):
    #         w = np.random.normal(0, scale=1, size=self.size)
    #         neighbors_list.append(self._project(matrix + w))
    #     return neighbors_list

    # In the relaxation, we have 0 <= var <= 1.
    def canonicalize(self):
        # HACK.
        X = Variable(*self.size)
        X.id = self.id
        constraints = [0 <= X, X <= 1, sum_entries(X, axis=1) == 1]
                       # diff((self.a.T*X).T) >= 0]

        canon_constr = []
        for cons in constraints:
            canon_constr += cons.canonical_form[1]
        return (X.canonical_form[0], canon_constr)
        # obj, constraints = super(Choose, self).canonicalize()
        # const_size = (self.size[1], 1)
        # mul_ones = lu.create_const(np.ones(const_size), const_size)
        # const_size = (self.size[0], 1)
        # rhs_ones = lu.create_const(np.ones(const_size), const_size)
        # lhs_expr = lu.rmul_expr(obj, mul_ones, const_size)
        # constraints += [lu.create_eq(mul_expr, constr_ones)]

        # # Ordering constraint.
        # aT_const = lu.create_const(self.a.T, (1, self.size[0]))
        # mul_expr = lu.mul_expr(aT_const, obj, (1, self.size[1]))
        # pos_neg_vec = lu.create_const( , (self.size[0], ))
