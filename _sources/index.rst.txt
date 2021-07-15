HPOflow Documentation
=====================

Tools for `Optuna <https://optuna.readthedocs.io/>`__,
`MLflow <https://www.mlflow.org/docs/latest/index.html>`__ and the integration of both.

The main part of this package is the :class:`~hpoflow.optuna_mlflow.OptunaMLflow` class.
It is used as a decorator for Optuna objective functions. If it is applied the Optuna
:class:`~optuna.study.Study` object is augmented.
This augmentation entails writing information to Optuna and MLflow in parallel.
Read more on the :ref:`documentation page <OptunaMLflow_doc>` and
the :ref:`code documentation page <optuna_mlflow_code_doc>`.

Content
=======

.. toctree::
   :glob:
   :maxdepth: 1

   doc
   code-doc
   license
   CONTRIBUTING
   CODE_OF_CONDUCT
   third-party-notices
   GitHub Repository <https://github.com/telekom/HPOflow>
   Imprint <https://www.telekom.com/imprint>


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
