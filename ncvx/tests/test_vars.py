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

import cvxpy as cp
from cvxpy import sum, Minimize, Maximize, Problem, Variable
from ncvx import *
import numpy as np
import unittest
from numpy.testing import assert_array_almost_equal


class TestVars(unittest.TestCase):
    """ Unit tests for the variable types. """
    def setUp(self):
        self._places = 5      # tune the tolerance down a bit as part of CVDOPT -> SCS switch
        self._solve = {'method': 'NC-ADMM', 'solver': cp.SCS, 'parallel': False}
        pass

    def test_boolean(self):
        """Test boolean variable."""
        x = Variable((5, 4))
        y = Boolean((5, 4))
        p = Problem(Minimize(sum(1 - x) + sum(x)), [x == y])
        result = p.solve(**self._solve)
        self.assertAlmostEqual(result[0], 20, places=self._places)
        for v in np.nditer(x.value):
            self.assertAlmostEqual(v * (1 - v), 0, places=self._places)

        # This time test a scalar variable, while restricting to a single entry
        # of a Boolean matrix.
        x = Variable()
        p = Problem(Minimize(sum(1 - x) + sum(x)), [x == Boolean((5, 4))[0, 0]])
        result = p.solve(**self._solve)
        self.assertAlmostEqual(result[0], 1, places=self._places)
        self.assertAlmostEqual(x.value * (1 - x.value), 0, places=self._places)

    def test_choose(self):
        """Test choose variable."""
        x = Variable((5, 4))
        y = Choose((5, 4), k=4)
        p = Problem(Minimize(sum(1 - x) + sum(x)), [x == y])
        result = p.solve(**self._solve)
        self.assertAlmostEqual(result[0], 20, places=self._places)
        for v in np.nditer(x.value):
            self.assertAlmostEqual(v * (1 - v), 0, places=self._places)
        self.assertAlmostEqual(x.value.sum(), 4, places=self._places)

    # Test card variable.
    def test_card(self):
        x = Card(5, k=3, M=1)
        p = Problem(Maximize(sum(x)), [x <= 1, x >= 0])
        result = p.solve(**self._solve)

        self.assertAlmostEqual(result[0], 3, places=self._places)
        for v in np.nditer(x.value):
            self.assertAlmostEqual(v * (1 - v), 0, places=self._places)
        self.assertAlmostEqual(x.value.sum(), 3, places=self._places)

        # Should be equivalent to x == choose.
        x = Variable((5, 4))
        c = Choose((5, 4), k=4)
        b = Boolean((5, 4))
        p = cp.Problem(Minimize(sum(1 - x) + sum(x)),
                       [x == c, x == b])
        result = p.solve(**self._solve)
        self.assertAlmostEqual(result[0], 20, places=self._places)
        for v in np.nditer(x.value):
            self.assertAlmostEqual(v * (1 - v), 0, places=self._places)

    # Test permutation variable.
    def test_permutation(self):
        x = Variable((1, 5))
        c = np.array([[1, 2, 3, 4, 5]])
        perm = Assign((5, 5))
        p = Problem(Minimize(sum(x)), [x == c*perm])
        result = p.solve(**self._solve)
        self.assertAlmostEqual(result[0], 15, places=self._places)
        assert_array_almost_equal(sorted(np.nditer(x.value)), c.ravel())
