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

class Annulus(NonCvxVariable):
    """ A variable satisfying r <= ||x||_2 <= R. """
    def __init__(self, rows, r, R, *args, **kwargs):
        self.r = r
        self.R = R
        assert 0 < r <= R
        super(Annulus, self).__init__(rows, 1, *args, **kwargs)

    def _project(self, matrix):
        if self.R >= norm(matrix, 2).value >= self.r:
            return matrix
        elif norm(matrix, 2).value == 0:
            result = np.ones(self.size)
            return self.r*result/norm(result, 2).value
        elif norm(matrix, 2).value < self.r:
            return self.r*matrix/norm(matrix, 2).value
        else:
            return self.R*matrix/norm(matrix, 2).value

    def _restrict(self, matrix):
        # Add restriction that beyond hyperplane at projection onto
        # n-sphere of radius r.
        return [matrix.T*self >= self.r*norm(matrix, 2).value]

    def canonicalize(self):
        obj, constr = super(Annulus, self).canonicalize()
        R = lu.create_const(self.R, (1, 1))
        return (obj, constr + [SOC(R, [obj])])
