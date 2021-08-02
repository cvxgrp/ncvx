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
import pytest
from cvxpy import sum, Minimize, Maximize, Problem, Variable
from ncvx import *
import numpy as np
from numpy.testing import assert_array_almost_equal

solve_args = {"method": "NC-ADMM", "solver": cp.SCS, "parallel": True, "verbose": False}


def approx(x, abs=1e-5, *args, **kwargs):
    return pytest.approx(x, abs=abs, *args, **kwargs)


def test_boolean():
    """Test boolean variable."""
    x = Variable((5, 4))
    y = Boolean((5, 4))
    p = Problem(Minimize(sum(1 - x) + sum(x)), [x == y])
    result = p.solve(**solve_args)
    assert result[0] == approx(20)
    for v in np.nditer(x.value):
        assert v * (1 - v) == approx(0)

    # This time test a scalar variable, while restricting to a single entry
    # of a Boolean matrix.
    x = Variable()
    p = Problem(Minimize(sum(1 - x) + sum(x)), [x == Boolean((5, 4))[0, 0]])
    result = p.solve(**solve_args)
    assert result[0] == approx(1)
    assert x.value * (1 - x.value) == approx(0)


def test_choose():
    """Test choose variable."""
    x = Variable((5, 4))
    y = Choose((5, 4), k=4)
    p = Problem(Minimize(sum(1 - x) + sum(x)), [x == y])
    result = p.solve(**solve_args)
    assert result[0] == approx(20)
    for v in np.nditer(x.value):
        assert v * (1 - v) == approx(0)
    assert x.value.sum() == approx(4)


def test_card():
    """Test card variable."""
    x = Card(5, k=3, M=1)
    p = Problem(Maximize(sum(x)), [x <= 1, x >= 0])
    result = p.solve(**solve_args)

    assert result[0] == approx(3)
    for v in np.nditer(x.value):
        assert v * (1 - v) == approx(0)
    assert x.value.sum() == approx(3)

    # Should be equivalent to x == choose.
    x = Variable((5, 4))
    c = Choose((5, 4), k=4)
    b = Boolean((5, 4))
    p = cp.Problem(Minimize(sum(1 - x) + sum(x)), [x == c, x == b])
    result = p.solve(**solve_args)
    assert result[0] == approx(20)
    for v in np.nditer(x.value):
        assert v * (1 - v) == approx(0)


def test_permutation():
    """Test permutation variable."""
    x = Variable((1, 5))
    c = np.array([[1, 2, 3, 4, 5]])
    perm = Assign((5, 5))
    p = Problem(Minimize(sum(x)), [x == c @ perm])
    result = p.solve(**solve_args)
    assert result[0] == approx(15)
    assert_array_almost_equal(sorted(np.nditer(x.value)), c.ravel())
