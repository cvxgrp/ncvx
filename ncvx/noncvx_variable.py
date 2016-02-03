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

import abc
import cvxpy
import cvxpy.interface as intf
import numpy as np

class NonCvxVariable(cvxpy.Variable):
    __metaclass__ = abc.ABCMeta
    def __init__(self, *args, **kwargs):
        super(NonCvxVariable, self).__init__(*args, **kwargs)
        self.noncvx = True
        self.z = cvxpy.Parameter(*self.size)
        self.u = cvxpy.Parameter(*self.size)
        self.u.value = np.zeros(self.size)

    def init_u(self, random=False):
        """Initializes the value of the dual variable.
        """
        self.u.value = np.zeros(self.size)

    # Verify that the matrix has the same dimensions as the variable.
    def validate_matrix(self, matrix):
        if self.size != intf.size(matrix):
            raise Exception(("The argument's dimensions must match "
                             "the variable's dimensions."))

    # Wrapper to validate matrix.
    def project(self, matrix):
        self.validate_matrix(matrix)
        return self._project(matrix)

    def dist(self, matrix):
        """Distance from matrix to projection.
        """
        proj_mat = self.project(matrix)
        return cvxpy.norm(cvxpy.vec(matrix - proj_mat), 2).value

    # Project the matrix into the space defined by the non-convex constraint.
    # Returns the updated matrix.
    @abc.abstractmethod
    def _project(matrix):
        return NotImplemented

    # Wrapper to validate matrix and update curvature.
    def restrict(self, matrix):
        matrix = self.project(matrix)
        return self._restrict(matrix)

    # Wrapper to validate matrix and update curvature.
    def neighbors(self, matrix):
        matrix = self.project(matrix)
        return self._neighbors(matrix)

    def relax(self):
        """The default convex relaxation.
        """
        return []

    # Fix the variable so it obeys the non-convex constraint.
    @abc.abstractmethod
    def _restrict(self, matrix):
        return NotImplemented
