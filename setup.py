from setuptools import setup

setup(
    name='ncvx',
    version='0.1.8',
    author='Steven Diamond, Reza Takapoui, Stephen Boyd',
     author_email='diamond@cs.stanford.edu, takapoui@stanford.edu, boyd@stanford.edu',
    packages=['ncvx'],
    license='GPLv3',
    zip_safe=False,
    install_requires=["cvxpy >= 0.3.5", "lap", "scsprox"],
    use_2to3=True,
    url='http://github.com/cvxgrp/ncvx/',
    description='A CVXPY extension for problems with convex objective and decision variables from a nonconvex set.',
)
