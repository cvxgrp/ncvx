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
import cvxpy.lin_ops.lin_utils as lu

class Choose(Boolean):
    """ A variable with k 1's and all other entries 0. """
    def __init__(self, rows=1, cols=1, k=None, *args, **kwargs):
        self.k = k
        super(Choose, self).__init__(rows, cols, *args, **kwargs)

    def init_z(self, random):
        """Initialize cloned variable.
        """
        super(Choose, self).init_z(random)
        self.z.value = self.k*self.z.value/self.z.value.sum()

    # The k-largest values are set to 1. The remainder are set to 0.
    def _project(self, matrix):
        indices = product(xrange(self.size[0]), xrange(self.size[1]))
        v_ind = sorted(indices, key=lambda ind: -matrix[ind])
        result = np.zeros(self.size)
        for ind in v_ind[0:self.k]:
            result[ind] = 1
        return result

    # In the relaxation, we have 0 <= var <= 1.
    def canonicalize(self):
        obj, constraints = super(Choose, self).canonicalize()
        k_const = lu.create_const(self.k, (1, 1))
        constraints += [lu.create_eq(lu.sum_entries(obj), k_const)]
        return (obj, constraints)

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
