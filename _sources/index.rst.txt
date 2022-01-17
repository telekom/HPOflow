HPOflow Documentation
=====================

Tools for `Optuna <https://optuna.readthedocs.io/>`__,
`MLflow <https://www.mlflow.org/docs/latest/index.html>`__ and the integration of both.

:class:`~hpoflow.optuna_mlflow.OptunaMLflow`
--------------------------------------------

The main part of this package is the :class:`~hpoflow.optuna_mlflow.OptunaMLflow` class.
It is used as a decorator for Optuna objective functions. If it is applied the Optuna
:class:`~optuna.study.Study` object is augmented.
This augmentation entails writing information to Optuna and MLflow in parallel.
Read more on the :ref:`OptunaMLflow documentation page <OptunaMLflow_doc>`.

:class:`~hpoflow.optuna_mlflow.OptunaMLflowCallback`
----------------------------------------------------
The :class:`~hpoflow.optuna_transformers.OptunaMLflowCallback` class integrates
`Optuna <https://optuna.readthedocs.io/>`__ and
`MLflow <https://www.mlflow.org/docs/latest/index.html>`__
with `Transformers <https://huggingface.co/transformers/>`__.
This is done by using :class:`~hpoflow.optuna_mlflow.OptunaMLflow` internally
and the :class:`transformers.TrainerCallback` to integrate with Transformers.
Read more on the :ref:`OptunaMLflowCallback documentation page <OptunaMLflowCallback_doc>`.

:class:`~hpoflow.optuna_mlflow.SignificanceRepeatedTrainingPruner`
------------------------------------------------------------------
This is an Optuna :mod:`Pruner <optuna.pruners>` which uses statistical significance as
an heuristic for decision-making. It prunes repeated trainings like in a cross validation.
As the test method a t-test is used.
Read more on the :ref:`SignificanceRepeatedTrainingPruner documentation page
<SignificanceRepeatedTrainingPruner_doc>`.

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
