HPOflow Documentation
=====================

.. image:: https://img.shields.io/badge/GitHub-Repository-lightgrey
   :alt: GitHub Repository
   :target: https://github.com/telekom/HPOflow
.. image:: https://img.shields.io/github/license/telekom/HPOflow
   :alt: MIT License
   :target: https://github.com/telekom/HPOflow/blob/main/LICENSE
.. image:: https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg
   :alt: Contributor Covenant v2.0
   :target: https://github.com/telekom/HPOflow/blob/main/CODE_OF_CONDUCT.md
.. image:: https://img.shields.io/pypi/pyversions/hpoflow
   :alt: Python Version
   :target: https://www.python.org
.. image:: https://img.shields.io/pypi/v/hpoflow.svg
   :alt: pypi
   :target: https://pypi.python.org/pypi/hpoflow
.. image:: https://github.com/telekom/HPOflow/actions/workflows/pytest.yml/badge.svg
   :alt: pytest status
   :target: https://github.com/telekom/HPOflow/actions/workflows/pytest.yml
.. image:: https://github.com/telekom/HPOflow/actions/workflows/static_checks.yml/badge.svg
   :alt: Static Code Checks status
   :target: https://github.com/telekom/HPOflow/actions/workflows/static_checks.yml
.. image:: https://github.com/telekom/HPOflow/actions/workflows/build_deploy_doc.yml/badge.svg
   :alt: Build & Deploy Doc
   :target: https://github.com/telekom/HPOflow/actions/workflows/build_deploy_doc.yml
.. image:: https://img.shields.io/github/issues-raw/telekom/HPOflow
   :alt: GitHub issues
   :target: https://github.com/telekom/HPOflow/issues

Tools for `Optuna <https://optuna.readthedocs.io/>`__,
`MLflow <https://www.mlflow.org/docs/latest/index.html>`__ and the integration of both.

:class:`~hpoflow.optuna_mlflow.OptunaMLflow`
--------------------------------------------

The main part of this package is the :class:`~hpoflow.optuna_mlflow.OptunaMLflow` class.
It is used as a decorator for Optuna objective functions. If it is applied the Optuna
:class:`~optuna.study.Study` object is augmented.
This augmentation entails writing information to Optuna and MLflow in parallel.
Read more on the :ref:`OptunaMLflow documentation page <OptunaMLflow_doc>`.

Other Components
----------------

- :ref:`SignificanceRepeatedTrainingPruner_doc`
- :ref:`OptunaMLflowCallback_doc`

Installation
------------

HPOflow is available at the `Python Package Index (PyPI) <https://pypi.org/project/hpoflow/>`__.
It can be installed with pip:

.. code-block:: bash

   $ pip install hpoflow

Some additional dependencies might be necessary.

To use :class:`hpoflow.optuna_mlflow.OptunaMLflow`:

.. code-block:: bash

   $ pip install mlflow GitPython

To use :class:`hpoflow.optuna_transformers.OptunaMLflowCallback`:

.. code-block:: bash

   $ pip install mlflow GitPython transformers

.. todo:
   add content

Content
-------

.. toctree::
   :glob:
   :maxdepth: 2

   doc
   code-doc
   License <https://github.com/telekom/HPOflow/blob/main/LICENSE>
   Contributing <https://github.com/telekom/HPOflow/blob/main/CONTRIBUTING.md>
   Contributor Covenant Code of Conduct <https://github.com/telekom/HPOflow/blob/main/CODE_OF_CONDUCT.md>
   GitHub Repository <https://github.com/telekom/HPOflow>
   Imprint <https://www.telekom.com/imprint>


Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
