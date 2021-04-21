# Copyright (c) 2021 Philip May
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import os

import setuptools

keywords = (
    "optuna mlflow deep-learning ml ai machine-learning experiment-tracking "
    "hyperparameter-optimization"
)

extras_require = {"checking": ["black", "flake8", "isort"], "optional": ["mlflow", "GitPython"]}


def get_version():
    version_filepath = os.path.join(os.path.dirname(__file__), "hpoflow", "version.py")
    with open(version_filepath) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.strip().split()[-1][1:-1]
    assert False


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="HPOflow",
    version=get_version(),
    maintainer="Philip May",
    author="Philip May",
    author_email="philip.may@t-systems.com",
    description="Tools for Optuna, MLflow and the integration of both",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/telekom/HPOflow",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "scipy",
        "optuna",
    ],
    extras_require=extras_require,
    keywords=keywords,
    classifiers=[
        "Development Status :: 3 - Alpha",
        # "Development Status :: 4 - Beta",
        # "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
    ],
)
