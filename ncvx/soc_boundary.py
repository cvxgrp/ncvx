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
from cvxpy.atoms.affine.index import index
from cvxpy.constraints.second_order import SOC
import numpy as np

class SOCBound(NonCvxVariable):
    """ A variable satisfying ||x||_2 == t. """
    def __init__(self, rows, *args, **kwargs):
        assert rows > 1
        super(SOCBound, self).__init__(rows, 1, *args, **kwargs)

    def init_z(self, random):
        """Initializes the value of the replicant variable.
        """
        if random:
            length = np.random.uniform()
            direction = np.random.randn(self.size[0])
            self.z.value = length*direction/norm(direction, 2).value
        else:
            self.z.value = np.zeros(self.size)

    # All values except k-largest (by magnitude) set to zero.
    def _project(self, matrix):
        result = np.zeros(self.size)
        last_entry = self.size[0]-1
        if np.all(matrix[0:last_entry] == 0):
            result[last_entry] = matrix[last_entry]
            result[0:last_entry] = result[last_entry,0]/np.sqrt(last_entry)
        else:
            matrix_norm = norm(matrix[0:last_entry], 2).value
            result[last_entry] = matrix[last_entry] + matrix_norm
            direction = matrix[0:last_entry]/matrix_norm
            result[0:last_entry] = result[last_entry,0]*direction
        return result

    # Constrain all entries to be the value in the matrix.
    def _restrict(self, matrix):
        return [self == matrix]

    def canonicalize(self):
        obj, constr = super(SOCBound, self).canonicalize()
        t = index.get_index(obj, constr, self.size[0]-1, 1)
        x = index.get_slice(obj, constr, 0, self.size[0]-1, 0, 1)
        return (obj, constr + [SOC(t, [x])])
