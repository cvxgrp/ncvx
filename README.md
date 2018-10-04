NCVX
====

NCVX is a package for modeling and solving problems with convex objectives and decision variables from a nonconvex set. This package provides heuristics such as NC-ADMM (a variation of alternating direction method of multipliers for nonconvex problems) and relax-round-polish, which can be viewed as a majorization-minimization algorithm. The solver methods provided and the syntax for constructing problems are discussed in [our associated paper](http://stanford.edu/~boyd/papers/ncvx.html).

NCVX is built on top of [CVXPY](http://www.cvxpy.org/), a domain-specific language for convex optimization embedded in Python.

Installation
------------
You should first install CVXPY version 0.4. CVXPY install guide can be found [here](http://www.cvxpy.org/). We are working on upgrading NCVX to work with CVXPY 1.0. 

Then install ``scsprox`` from source [here](https://github.com/SteveDiamond/scsprox).

The easiest way to install the package is to run ``pip install ncvx``. To install the package from source, run ``python setup.py install`` in the main folder. The package has CVXPY and lap as dependencies.

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
prob.solve(method="NC-ADMM")
print objective.value
print x.value
```
Other solve methods can be used by simply changing the solve method, for example ``prob.solve(method="relax-round-polish")`` uses relax-round-polish to approximately solve the problem. Constraints can be added to the problem similar to CVXPY. For example, the following code approximately solves the above problem, with the additional constraint that the components of ``x`` must add up to zero.
```
x = Card(n, k, M)
objective = sum_squares(A*x - b)
constraints = [sum(x) == 0]
prob = Problem(Minimize(objective), constraints)
prob.solve(method="NC-ADMM")
print objective.value
print x.value
```

Variable constructors
---------------------
The following sets are supported by NCVX at the moment:
* ``Boolean(n)`` creates a variable with ``n`` Boolean components.
* ``integer(n, M)`` creates a variable with ``n`` integer components between ``-M`` and ``M``.
* ``Card(n, k, M)`` creates a variable with ``n`` components, with the implicit constraint that at most ``k`` entries are nonzero and all entries are between ``-M`` and ``M``.
* ``Choose(n, k)`` creates a variable with ``n`` Boolean components, with the implicit constraint that it has exactly ``k`` nonozero entries.
* ``Annulus(n, r, R)`` creates a variable with ``n`` components with the implicit constraint that its Euclidean norm is between ``r`` and ``R``.
* ``Sphere(n, r)`` creates a variable with ``n`` components with the implicit constraint that its Euclidean norm is equal to ``r``.
* ``Rank(m, n, k, M)`` creates a ``m x n`` matrix variable with the implicit constraints that its rank is at most ``k`` and its Euclidean norm is at most ``M``.
* ``Assign(m, n) `` creates a ``m x n`` matrix variable with the implicit constraint that it is an assignment matrix.
* ``Permute(n)`` creates a ``n x n`` matrix variable with the implicit constraint that it is a permutation matrix.
* ``Cycle(n)`` creates a ``n x n`` matrix variable with the implicit constraint that it is the adjacency matrix of a Hamiltonian cycle.


Users can extend these heuristics to additional problems by adding other sets. Each set must support a method for (approximate) projection. We do not require but benefit from knowing a convex relaxation of the set, a convex restriction at any point in the set, and the neighbors of any point in the set under some discrete distance metric.

Variable methods
----------------
Each variable supports the following methods:
* ``variable.relax()`` returns a list of convex constraints that represent a convex relaxation of the nonconvex set, to which the variable belongs.
* ``variable.project(z)`` returns the Euclidean (or approximate) projection of ``z`` onto the nonconvex set C, to which the variable belongs.
* ``variable.restrict(z)`` returns a list of convex constraints describing the convex restriction at ``z`` of the nonconvex set, to which the variable belongs.
* ``variable.neighbors(z)`` returns a list of neighbors of ``z`` contained in the nonconvex set, to which the variable belongs.

Constructing and solving problems
---------------------------------
The components of the variable, the objective, and the constraints are constructed using standard CVXPY syntax. Once the user has constructed a problem object, they can apply the following solve methods:
* ``problem.solve(method="relax")`` solves the convex relaxation of the problem.
* ``problem.solve(method="relax-round-polish")`` applies the relax-round-polish heuristic. Additional arguments can be used to specify the parameters.
* ``problem.solve(method="NC-ADMM")`` applies the NC-ADMM heuristic. Additional arguments can be used to specify the number of starting points and the number of iterations the algorithm is run from each starting point.
