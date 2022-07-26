from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    extras_require={
        "dev": [
            "flake8",
            "mypy",
            "commitizen",
            "pre-commit",
            "pipenv-setup[black]",
            "twine",
            "build",
        ]
    },
    name="PyFixedReps-andnp",
    url="https://github.com/andnp/PyFixedReps.git",
    author="Andy Patterson",
    author_email="andnpatterson@gmail.com",
    packages=find_packages(exclude=["tests*"]),
    version="2.0.0",
    license="MIT",
    description="A small set of fixed representations usually used in RL",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    install_requires=["numba>=0.52.0", "numpy>=1.21.0"],
)
