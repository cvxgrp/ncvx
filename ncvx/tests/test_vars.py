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

from cvxpy import *
from ncvx import *
import numpy as np
import unittest
from numpy.testing import assert_array_almost_equal
from builtins import range


class TestVars(unittest.TestCase):
    """ Unit tests for the variable types. """
    def setUp(self):
        pass

    # Test boolean variable.
    def test_boolean(self):
        x = Variable((5, 4))
        y = Boolean(5, 4)
        p = Problem(Minimize(sum(1-x) + sum(x)), [x == y])
        p = Problem(Minimize(sum_entries(1-x) + sum_entries(x)), [x == y])
        result = p.solve(method="NC-ADMM", solver=CVXOPT)
        self.assertAlmostEqual(result[0], 20)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                v = x.value[i, j]
                self.assertAlmostEqual(v*(1-v), 0)

        x = Variable()
        p = Problem(Minimize(sum(1-x) + sum(x)), [x == Boolean(5,4)[0,0]])
        p = Problem(Minimize(sum_entries(1-x) + sum_entries(x)), [x == Boolean(5,4)[0,0]])
        result = p.solve(method="NC-ADMM", solver=CVXOPT)
        self.assertAlmostEqual(result[0], 1)
        self.assertAlmostEqual(x.value*(1-x.value), 0)

    # Test choose variable.
    def test_choose(self):
        x = Variable((5, 4))
        y = Choose(5, 4, k=4)
        p = Problem(Minimize(sum(1-x) + sum(x)),
                    [x == y])
        result = p.solve(method="NC-ADMM", solver=CVXOPT)
        self.assertAlmostEqual(result[0], 20)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                v = x.value[i, j]
                self.assertAlmostEqual(v*(1-v), 0)
        self.assertAlmostEqual(x.value.sum(), 4)

    # Test card variable.
    def test_card(self):
        x = Card(5, k=3, M=1)
        p = Problem(Maximize(sum(x)),
            [x <= 1, x >= 0])
        result = p.solve(method="NC-ADMM")
        self.assertAlmostEqual(result[0], 3)
        for v in np.nditer(x.value):
            self.assertAlmostEqual(v*(1-v), 0)
        self.assertAlmostEqual(x.value.sum(), 3)

        #should be equivalent to x == choose
        x = Variable((5, 4))
        c = Choose(5, 4, k=4)
        b = Boolean(5, 4)
        p = Problem(Minimize(sum(1-x) + sum(x)),
                    [x == c, x == b])
        result = p.solve(method="NC-ADMM", solver=CVXOPT)
        self.assertAlmostEqual(result[0], 20)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                v = x.value[i, j]
                self.assertAlmostEqual(v*(1-v), 0)

    # Test permutation variable.
    def test_permutation(self):
        x = Variable((1, 5))
        c = np.array([[1,2,3,4,5]])
        perm = Assign(5, 5)
        p = Problem(Minimize(sum(x)), [x == c*perm])
        p = Problem(Minimize(sum_entries(x)), [x == c*perm])
        result = p.solve(method="NC-ADMM")
        self.assertAlmostEqual(result[0], 15)
        assert_array_almost_equal(sorted(np.nditer(x.value)), c.ravel())