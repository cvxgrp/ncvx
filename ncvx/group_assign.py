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
import cvxpy as cvx
import lap
import numpy as np


class GroupAssign(Boolean):
    """ A group assignment matrix.

    This is a special case (w_ij = 1) of generalized assignment problem.
    (See https://en.wikipedia.org/wiki/Generalized_assignment_problem for GAP.)

    Assign s_j people to jth group.

    Here the set X is size of (m x n), where m is the number of people and
     n is the number of groups. Also, m >= n.

    The set is:
        sum_j X_ij = 1
        sum_i X_ij = s_j
        X_ij \in {0, 1}
    """
    def __init__(self, rows, cols, col_sum, *args, **kwargs):
        assert rows >= cols
        assert rows == sum(col_sum)
        super(GroupAssign, self).__init__(rows=rows, cols=cols, *args, **kwargs)
        self.col_sum = col_sum

    def init_z(self, random):
        if random:
            result = np.zeros(self.size)
            num_entries = self.size[0]*self.size[1]
            weights = np.random.uniform(size=num_entries)
            weights /= weights.sum()
            for k in range(num_entries):
                assignment = np.random.permutation(self.size[0])
                for j in range(self.size[1]):
                    result[assignment[j], j] += weights[k]
            self.z.value = result
        else:
            self.z.value = np.ones(self.size)/self.size[1]

    # Compute projection with maximal weighted matching.
    def _project(self, matrix):
        if self.is_scalar():
            return 1
        else:
            # Note that we use Munkres algorithm, but expand columns from n to m
            # by replicating each column by group size.
            mm = np.repeat(matrix, self.col_sum, axis=1)
            indexes = lap.lapjv(np.asarray(-mm))
            result = np.zeros(self.size)
            reduce = np.repeat(range(len(self.col_sum)), self.col_sum)
            for row, column in enumerate(indexes[1]):
                # map expanded column index to reduced group index.
                result[row, reduce[column]] = 1
            return result

    # Constrain all entries to be zero that correspond to
    # zeros in the matrix.
    def _restrict(self, matrix):
        return [self == matrix]

    def _neighbors(self, matrix):
        """Neighbors swap adjacent rows.

        """
        neighbors_list = []
        for i in range(self.size[0]-1):
            # Add to neighbor only when the candidate person (row) is in a different group.
            new_mat = matrix.copy()
            for j in range(i+1, self.size[0]-1):
                if np.all(matrix[i, :] == matrix[j, :]):
                    continue
                else:
                    new_mat[j,:] = matrix[i,:]
                    new_mat[i,:] = matrix[j,:]
                    neighbors_list += [new_mat]
                    break
        return neighbors_list

    def relax(self):
        """Convex relaxation.
        """
        constr = super(GroupAssign, self).relax()
        return constr + [
            cvx.sum_entries(self, axis=1) == 1,
            cvx.sum_entries(self, axis=0) == self.col_sum[np.newaxis, :]
        ]
