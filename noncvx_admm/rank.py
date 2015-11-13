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
from cvxpy import Variable
import numpy as np

# TODO nuclear norm as convex relaxation?

class Rank(NonCvxVariable):
    """ A variable satisfying Rank(X) <= k. """
    def __init__(self, rows, cols, k, M=None, symmetric=False, *args, **kwargs):
        self.k = k
        self.M = M
        self.symmetric = symmetric
        super(Rank, self).__init__(rows, cols, *args, **kwargs)

    def init_z(self, random):
        """Initializes the value of the replicant variable.
        """
        if random:
            self.z.value = np.random.uniform(0, self.sigma, size=self.size)
        else:
            self.z.value = np.zeros(self.size)

    def _project(self, matrix):
        """All singular values except k-largest (by magnitude) set to zero.
        """
        if self.symmetric:
            w, V = np.linalg.eigh(matrix)
            w_sorted_idxs = np.argsort(-w)
            w[w_sorted_idxs[self.k:]] = 0
            return V.dot(np.diag(w)).dot(V.T)
        else:
            U, s, V = np.linalg.svd(matrix)
            s[self.k:] = 0
            return U.dot(np.diag(s)).dot(V)

    # Constrain all entries to be the value in the matrix.
    def _restrict(self, matrix):
        # TODO is this really working?
        if self.symmetric:
            w, V = np.linalg.eigh(matrix)
            w_sorted_idxs = np.argsort(-w)
            pos_w = w[w_sorted_idxs[:self.k]]
            pos_V = V[:,w_sorted_idxs[:self.k]]
            # print V.dot(np.diag(w)).dot(V.T) - pos_V.dot(np.diag(pos_w)).dot(pos_V.T)
            Sigma = Variable(self.k, self.k)
            return [self == pos_V*Sigma*pos_V.T]
        else:
            U, s, V = np.linalg.svd(matrix)
            Sigma = Variable(self.k, self.k)
            return [self == U[:,:self.k]*Sigma*V.T[:self.k,:]]

    # def canonicalize(self):
    #     norm(self, 2) <= self.M
