.. _SignificanceRepeatedTrainingPruner_doc:

:class:`~hpoflow.optuna.SignificanceRepeatedTrainingPruner`
===========================================================

.. seealso::
   Code documentation can be found here: :ref:`code documentation page <optuna_code_doc>`

This is an Optuna :mod:`Pruner <optuna.pruners>` which uses statistical significance as
an heuristic for decision-making. It prunes repeated trainings like in a cross validation.
As the test method a t-test is used.

.. todo::
   - add more details
   - add example
