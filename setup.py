# Copyright (c) 2021 Philip May
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Build script for setuptools."""

import os

import setuptools


project_name = "hpoflow"
source_code = "https://github.com/telekom/HPOflow"
keywords = (
    "optuna mlflow deep-learning ml ai machine-learning experiment-tracking "
    "hyperparameter-optimization"
)
install_requires = [
    "numpy",
    "scipy",
    "optuna",
]
extras_require = {
    "checking": [
        "black",
        "flake8",
        "isort",
        "mdformat",
        "pydocstyle",
        "mypy",
        "pylint",
        "pylintfileheader",
    ],
    "optional": ["mlflow", "GitPython", "transformers"],
    "testing": ["pytest", "scikit-learn", "torch"],
    "doc": ["sphinx", "sphinx_rtd_theme", "recommonmark", "sphinx_copybutton"],
}


def get_version():
    """Read version from ``version.py``."""
    version_filepath = os.path.join(os.path.dirname(__file__), project_name, "version.py")
    with open(version_filepath) as f:
        for line in f:
            if line.startswith("__version__"):
                return line.strip().split()[-1][1:-1]
    assert False


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name=project_name,
    version=get_version(),
    maintainer="Philip May",
    author="Philip May",
    author_email="philip.may@t-systems.com",
    description="Tools for Optuna, MLflow and the integration of both",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=source_code,
    project_urls={
        "Bug Tracker": source_code + "/issues",
        "Documentation": "https://telekom.github.io/HPOflow/",
        "Source Code": source_code,
        "Contributing": "https://github.com/telekom/HPOflow/blob/main/CONTRIBUTING.md",
        "Code of Conduct": "https://github.com/telekom/HPOflow/blob/main/CODE_OF_CONDUCT.md",
    },
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=install_requires,
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
        # "Programming Language :: Python :: 3.6",
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
