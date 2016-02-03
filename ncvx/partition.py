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
import cvxpy as cvx

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

    def relax(self):
        """Convex relaxation.
        """
        constr = super(Partition, self).relax()
        return constr + [cvx.sum_entries(self, axis=1) == 1]
