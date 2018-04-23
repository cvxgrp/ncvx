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
import cvxpy.lin_ops.lin_utils as lu
import numpy as np
import lap


class Assign(Boolean):
    """ An assignment matrix.

        Assign jobs x workers. Assign one worker to each job.
    """
    def __init__(self, rows, cols, *args, **kwargs):
        assert rows >= cols
        super(Assign, self).__init__(rows=rows, cols=cols, *args, **kwargs)

    def init_z(self, random):
        if random:
            # Random relaxation of assignment matrix.
            # convex combination of mn assignment matrices.
            # This is a distribution over all relaxations.
            # http://planetmath.org/proofofbirkhoffvonneumanntheorem
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
            indexes = lap.lapjv(np.asarray(-matrix))
            result = np.zeros(self.size)
            for row, column in enumerate(indexes[1]):
                result[row, column] = 1
            return result

    def matrix_to_lists(self, matrix):
        """Convert a matrix to a list of lists.
        """
        rows, cols = matrix.shape
        lists = []
        for i in range(rows):
            lists.append(matrix[i,:].tolist()[0])
        return lists

    # Constrain all entries to be zero that correspond to
    # zeros in the matrix.
    def _restrict(self, matrix):
        return [self == matrix]


    def _neighbors(self, matrix):
        """Neighbors swap adjacent rows.
        """
        neighbors_list = []
        for i in range(self.size[0]-1):
            new_mat = matrix.copy()
            new_mat[i+1,:] = matrix[i,:]
            new_mat[i,:] = matrix[i+1,:]
            neighbors_list += [new_mat]
        return neighbors_list

    # In the relaxation, we have 0 <= var <= 1.
    def canonicalize(self):
        obj, constraints = super(Assign, self).canonicalize()
        shape = (self.size[1], 1)
        one_row_vec = lu.create_const(np.ones(shape), shape)
        shape = (1, self.size[0])
        one_col_vec = lu.create_const(np.ones(shape), shape)
        # Row sum <= 1
        row_sum = lu.rmul_expr(obj, one_row_vec, (self.size[0], 1))
        constraints += [lu.create_leq(row_sum, lu.transpose(one_col_vec))]
        # Col sum == 1.
        col_sum = lu.mul_expr(one_col_vec, obj, (1, self.size[1]))
        constraints += [lu.create_eq(col_sum, lu.transpose(one_row_vec))]
        return (obj, constraints)
