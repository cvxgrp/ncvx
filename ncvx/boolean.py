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
import cvxpy.lin_ops.lin_utils as lu
from cvxpy import norm
import numpy as np

class Boolean(NonCvxVariable):
    """ A boolean variable. """
    # Sets the initial z value to a matrix of 0.5's.
    def init_z(self, random):
        if random:
            self.z.value = np.random.uniform(size=self.size)
        else:
            self.z.value = np.zeros(self.size) + 0.5

    # All values set rounded to zero or 1.
    def _project(self, matrix):
        return np.around(matrix) #> 0

    # Constrain all entries to be the value in the matrix.
    def _restrict(self, matrix):
        # return [self == matrix]
        return [norm(self - matrix, 1) <= 1]

    def _neighbors(self, matrix):
        neighbors_list = []
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                new_mat = matrix.copy()
                new_mat[i,j] = 1 - new_mat[i,j]
                neighbors_list += [new_mat]
        return neighbors_list

    def relax(self):
        """The convex relaxation.
        """
        constr = super(Boolean, self).relax()
        return constr + [0 <= self, self <= 1]
