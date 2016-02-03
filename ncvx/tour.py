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

from .assign import Assign
import numpy as np
import cvxpy as cvx

# TODO change to cycle.

class Tour(Assign):
    """ A permutation matrix that describes a single cycle.
        e.g. 1->3->5->2->4->1
    """
    def __init__(self, n, *args, **kwargs):
        super(Tour, self).__init__(rows=n, cols=n, *args, **kwargs)

    # Compute projection with maximal weighted matching.
    def _project(self, matrix):
        if self.is_scalar():
            return 1
        else:
            matrix = matrix.copy()
            # Greedy algorithm.
            # Fix largest entry that still could be a tour.
            # Recurse.
            tour = np.zeros(self.size[0]) - 1
            result = np.zeros(self.size)
            for i in range(self.size[0]):
                while True:
                    idx = np.argmax(matrix)
                    row, col = zip(*np.unravel_index([idx], self.size))[0]
                    # Check that consistent with tour.
                    tour[row] = col
                    if self._no_cycles(tour):
                        result[row, col] = 1
                        matrix[row,:] = -np.inf
                        matrix[:,col] = -np.inf
                        break
                    else:
                        matrix[row, col] = -np.inf
                        tour[row] = -1
                    assert not (matrix == -np.inf).all()
            return result

    def _no_cycles(self, tour):
        """Return true if the tour has no cycles.
        """
        for i in range(self.size[0]):
            visited = []
            cur = i
            while True:
                visited.append(cur)
                cur = tour[cur]
                if cur in visited:
                    return len(visited) == self.size[0]
                elif cur == -1:
                    break
        return True

    def _neighbors(self, matrix):
        """Swap a->b->c->d to a->c->b->d
        """
        neighbors_list = []
        idxs = np.argmax(matrix, axis=1)
        for a in range(self.size[0]):
            new_mat = matrix.copy()
            b = idxs[a]
            c = idxs[b]
            d = idxs[c]

            new_mat[a,c] = 1
            new_mat[a,b] = 0

            new_mat[b,d] = 1
            new_mat[b,c] = 0

            new_mat[c,b] = 1
            new_mat[c,d] = 0

            neighbors_list += [new_mat]
        return neighbors_list

    def relax(self):
        """Convex relaxation.
        """
        constr = super(Tour, self).relax()
        # Ensure it's a tour.
        n = self.size[0]
        # Diagonal == 0 constraint.
        constr += [cvx.diag(self) == 0]
        # Spectral constraint.
        mat_val = np.cos(2*np.pi/n)*np.eye(n) + 4*np.ones((n,n))/n
        return constr + [mat_val >> (self + self.T)/2]
