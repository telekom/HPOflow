.. _SignificanceRepeatedTrainingPruner_doc:

:class:`~hpoflow.optuna.SignificanceRepeatedTrainingPruner`
===========================================================

.. seealso::
   Code documentation can be found here: :ref:`code documentation page <optuna_code_doc>`

This is an Optuna :mod:`Pruner <optuna.pruners>` which uses statistical significance as
an heuristic for decision-making. It prunes repeated trainings like in a cross validation.
As the test method a t-test is used.

:mod:`Optuna's standard pruners <optuna.pruners>` assume that you only adjust the model once per
hyperparameter set. Those pruners work on the basis of intermediate results. For example, once per
epoch. In contrast, this pruner does not work on intermediate results but on the results of a
cross validation or more precisely the results of the individual folds.

.. todo::
   - add more details
   - add example
