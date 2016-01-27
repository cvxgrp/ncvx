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
from cvxpy import norm
from cvxpy.constraints.second_order import SOC
import cvxpy.lin_ops.lin_utils as lu
import numpy as np

class Sphere(NonCvxVariable):
    """ A variable satisfying ||x||_2 = 1. """
    def __init__(self, rows=1, *args, **kwargs):
        super(Sphere, self).__init__(rows, 1, *args, **kwargs)

    def init_z(self, random):
        """Initializes the value of the replicant variable.
        """
        if random:
            length = np.random.uniform()
            direction = np.random.randn(self.size)
            self.z.value = length*direction/norm(direction, 2).value
        else:
            self.z.value = np.zeros(self.size)

    def _project(self, matrix):
        if np.all(matrix == 0):
            result = np.ones(self.size)
            return result/norm(result, 2).value
        else:
            return matrix/norm(matrix, 2).value

    # Constrain all entries to be the value in the matrix.
    def _restrict(self, matrix):
        return [self == matrix]

    def canonicalize(self):
        obj, constr = super(Sphere, self).canonicalize()
        one = lu.create_const(1, (1, 1))
        return (obj, constr + [SOC(one, [obj])])
