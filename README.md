NCVX
====

NCVX is a package for modeling and solving problems with convex objectives and decision variables from a nonconvex set.
The solver methods provided and the syntax for constructing problems are discussed in [our associated paper](http://stanford.edu/~boyd/papers/ncvx.html).

NCVX is built on top of [CVXPY](http://www.cvxpy.org/),
a domain-specific language for convex optimization embedded in Python.

Installation
------------

Make sure you have the latest version of CVXPY by running ``pip install —upgrade cvxpy``. The easiest way to install the package is to run ``pip install ncvx``.

To install the package from source, run ``python setup.py install`` in the main folder.
The package has CVXPY and munkres as dependencies.

Example
-------
The following code uses NC-ADMM heuristic to approximately solve a least-squares problem where the variable has only ``k`` nonzero components, and all components are between -1 and 1.
```
# Problem data
m = 30; n = 20; k = 6
numpy.random.seed(1)
A = numpy.random.randn(m, n)
b = numpy.random.randn(m)

#NC-ADMM heuristic.
x = Card(n, k, 1)
objective = sum_squares(A*x - b)
prob = Problem(Minimize(objective), [])
prob.solve(method="admm")
print objective.value
print x.value
```
Other solve methods can be used by simply changing the solve method, for example ``prob.solve(method="relax_round_polish")`` uses relax-round-polish to approximately solve the problem. Constraints can be added to the problem similar to CVXPY. For example, the following code approximately solves the above problem, with the additional constraint that the components of ``x`` must add up to zero.
```
 = Card(n, k, M)
objective = sum_squares(A*x - b)
constraints = [sum(x) == 0]
prob = Problem(Minimize(objective), constraints)
prob.solve(method="admm")
print objective.value
print x.value
```
