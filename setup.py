from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='PyFixedReps-andnp',
    url='https://github.com/andnp/PyFixedReps.git',
    author='Andy Patterson',
    author_email='andnpatterson@gmail.com',
    packages=find_packages(exclude=['tests*']),
    version='0.0.5',
    license='MIT',
    description='A small set of fixed representations usually used in RL',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    install_requires=[ 'numpy', 'numba' ],
)
