from setuptools import setup

setup(
    name='noncvx_admm',
    version='0.2.17',
    author='Steven Diamond, Reza Takapoui, Stephen Boyd',
    packages=['noncvx_admm'],
    license='GPLv3',
    zip_safe=False,
    install_requires=["cvxpy >= 0.2.16", "munkres"],
    use_2to3=True,
)
