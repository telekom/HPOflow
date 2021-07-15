.. _OptunaMLflowCallback_doc:

:class:`~hpoflow.optuna_transformers.OptunaMLflowCallback`
==========================================================

.. seealso::
   Code documentation can be found here:
   :ref:`code documentation page <optuna_transformers_code_doc>`

The :class:`~hpoflow.optuna_transformers.OptunaMLflowCallback` class integrates
`Optuna <https://optuna.readthedocs.io/>`__ and
`MLflow <https://www.mlflow.org/docs/latest/index.html>`__
with `Transformers <https://huggingface.co/transformers/>`__.
This is done by using :class:`~hpoflow.optuna_mlflow.OptunaMLflow` internally
and :class:`transformers.TrainerCallback` to integrate with Transformers.

.. todo::
   - add more details
   - add example
