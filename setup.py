from setuptools import setup, find_packages

setup(
    name='PyFixedReps',
    url='https://github.com/andnp/PyFixedReps.git',
    author='Andy Patterson',
    author_email='andnpatterson@gmail.com',
    packages=find_packages(exclude=['tests*']),
    install_requires=[ 'numpy', 'numba' ],
    version=0.4,
    license='MIT',
    description='A small set of fixed representations usually used in RL',
    long_description='todo',
)
