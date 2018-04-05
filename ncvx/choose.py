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
import numpy as np
import cvxpy as cvx

class Choose(Boolean):
    """ A variable with k 1's and all other entries 0. """
    def __init__(self, rows=1, cols=1, k=None, *args, **kwargs):
        if k is None:
            raise Exception("Choose requires a value for k.")
        self.k = k
        super(Choose, self).__init__(rows, cols, *args, **kwargs)

    def init_z(self, random):
        """Initialize cloned variable.
        """
        super(Choose, self).init_z(random)
        self.z.value = self.k*self.z.value/self.z.value.sum()

    # The k-largest values are set to 1. The remainder are set to 0.
    def _project(self, matrix):
        lin_index = np.squeeze(np.asarray(matrix)).flatten().argsort()[::-1]
        sub_index = np.unravel_index(lin_index[:self.k], matrix.shape)
        matrix[:] = 0.0
        matrix[sub_index] = 1.0
        return matrix

    # In the relaxation, we have 0 <= var <= 1.
    def relax(self):
        constr = super(Choose, self).relax()
        constr += [cvx.sum_entries(self) == self.k]
        return constr

    def _neighbors(self, matrix):
        # Can swap a 1 with a neighboring 0.
        neighbors_list = []
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if matrix[i,j] == 1:
                    for k in range(i-1,i+1):
                        for l in range(j-1,j+1):
                            if k != i and l != j and \
                               0 <= k < self.size[0] and \
                               0 <= l < self.size[1] and \
                               matrix[k,l] == 0:
                               new_mat = matrix.copy()
                               new_mat[i,j] = 0
                               new_mat[k,l] = 1
                               neighbors_list += [new_mat]

        return neighbors_list
