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

from .noncvx_variable import NonCvxVariable
import cvxpy.interface.matrix_utilities as intf
from itertools import product
import numpy as np
import cvxpy as cvx

class Card(NonCvxVariable):
    """ A variable with constrained cardinality. """
    # k - the maximum cardinality of the variable.
    # M - upper bound on ||x||_inf
    def __init__(self, rows, k, M, *args, **kwargs):
        self.k = k
        self.M = M
        super(Card, self).__init__(rows, 1, *args, **kwargs)

    def init_z(self, random):
        """Initializes the value of the replicant variable.
        """
        if random:
            alpha = np.random.uniform(0, self.k*self.M)
            y = np.random.uniform(-self.M, self.M, size=self.size)
            self.z.value = y*alpha/np.abs(y).sum()
        else:
            self.z.value = np.zeros(self.size)

    # All values except k-largest (by magnitude) set to zero.
    def _project(self, matrix):
        indices = product(list(range(self.size[0])), list(range(self.size[1])))
        v_ind = sorted(indices, key=lambda ind: -abs(matrix[ind]))
        result = matrix.copy()
        for ind in v_ind[self.k:]:
           result[ind] = 0
        return np.maximum(-self.M, np.minimum(result, self.M))

    # Constrain all entries to be zero that correspond to
    # zeros in the matrix.
    def _restrict(self, matrix):
        constraints = []
        rows, cols = intf.size(matrix)
        for i in range(rows):
            for j in range(cols):
                if matrix[i, j] == 0:
                    constraints.append(self[i, j] == 0)
        return constraints

    def relax(self):
        """The convex relaxation.
        """
        constr = super(Card, self).relax()
        return constr + [cvx.norm(self, 1) <= self.k*self.M,
                         cvx.norm(self, 'inf') <= self.M]
