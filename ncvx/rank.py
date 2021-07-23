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
import cvxpy as cp
import numpy as np


def Rank(shape, k, M=None, symmetric=False):
    if symmetric:
        return SymmRank(shape, k, M)
    else:
        return AsymmRank(shape, k, M)


class AsymmRank(NonCvxVariable):
    """ A variable satisfying Rank(X) <= k. """
    def __init__(self, shape, k, M, *args, **kwargs):
        self.k = k
        self.M = M
        super().__init__(shape, *args, **kwargs)

    def init_z(self, random):
        """Initializes the value of the replicant variable.
        """
        if random:
            self.z.value = np.random.uniform(0, self.sigma, size=self.shape)
        else:
            self.z.value = np.zeros(self.shape)

    def _project(self, matrix):
        """All singular values except k-largest (by magnitude) set to zero.
        """
        U, s, V = np.linalg.svd(matrix)
        s[self.k:] = 0
        return U.dot(np.diag(s)).dot(V)

    def _restrict(self, matrix):
        U, s, V = np.linalg.svd(matrix)
        Sigma = cp.Variable(self.k, self.k)
        return [self == U[:, :self.k] * Sigma*V.T[:self.k, :]]

    def relax(self):
        if self.M is None:
            return []
        else:
            return [cp.norm(self, 2) <= self.M]


class SymmRank(AsymmRank):
    """ A symmetric variable satisfying Rank(X) <= k. """

    def _project(self, matrix):
        """All singular values except k-largest (by magnitude) set to zero.
        """
        w, V = np.linalg.eigh(matrix)
        w_sorted_idxs = np.argsort(-w)
        w[w_sorted_idxs[self.k:]] = 0
        return V.dot(np.diag(w)).dot(V.T)

    # Constrain all entries to be the value in the matrix.
    def _restrict(self, matrix):
        w, V = np.linalg.eigh(matrix)
        w_sorted_idxs = np.argsort(-w)
        pos_w = w[w_sorted_idxs[:self.k]]
        pos_V = V[:, w_sorted_idxs[:self.k]]
        Sigma = cp.Symmetric(self.k, self.k)
        return [self == pos_V*Sigma*pos_V.T]

    def relax(self):
        return super().relax() + [self == self.T]
