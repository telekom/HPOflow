.. _OptunaMLflow_doc:

:class:`~hpoflow.optuna_mlflow.OptunaMLflow`
============================================

.. seealso::
   Code documentation can be found here: :ref:`code documentation page <optuna_mlflow_code_doc>`

The :class:`~hpoflow.optuna_mlflow.OptunaMLflow` class is used as a decorator for
`Optuna <https://optuna.readthedocs.io/>`__ objective functions. It looks like this:

.. code-block:: python
   :emphasize-lines: 1

   @OptunaMLflow()
   def objective(trial):
       x = trial.suggest_float("x", -10, 10)
       return (x - 2) ** 2

   study = optuna.create_study()
   study.optimize(objective, n_trials=100)

If this decorator is applied the Optuna :class:`optuna.study.Study` object is augmented.
This augmentation entails logging information to `Optuna <https://optuna.readthedocs.io/>`__
and `MLflow <https://www.mlflow.org/docs/latest/index.html>`__ in parallel.

Autologging to MLflow
---------------------

You can use Optuna and the Trial object as you are used to.
The following is automatically logged in parallel:

* the Optuna distributions and parameters (name and ranges)
* the sampled hyperparameters
* the objective result
* exceptions with traceback
* the direction of the study (``MINIMIZE`` or ``MAXIMIZE``)
* hostname and process id

Manual Logging
--------------

In addition to what is logged automatically by Optuna usage,
the following can be logged manually:

* additonal metrics: :meth:`~hpoflow.optuna_mlflow.OptunaMLflow.log_metric` or
  :meth:`~hpoflow.optuna_mlflow.OptunaMLflow.log_metrics`
* parameter: :meth:`~hpoflow.optuna_mlflow.OptunaMLflow.log_param` or
  :meth:`~hpoflow.optuna_mlflow.OptunaMLflow.log_params`
* tags: :meth:`~hpoflow.optuna_mlflow.OptunaMLflow.set_tag` or
  :meth:`~hpoflow.optuna_mlflow.OptunaMLflow.set_tags`

.. note::
   The manually logged information is stored on the Optuna side as a user attribute on the
   :class:`~optuna.trial.Trial` object (also see :meth:`~optuna.trial.Trial.set_user_attr`).

Logging Nested Runs
-------------------

Sometimes you want to repeat a training several times with the same hyperparameters within a trial.
This is the case, for example, when performing a cross-validation.
It is possible to log the results of these repetitions as so-called nested runs on the MLflow side.
To do this use the :meth:`~hpoflow.optuna_mlflow.OptunaMLflow.log_iter` method.
It looks like this:

.. code-block:: python
   :emphasize-lines: 9

   @OptunaMLflow()
   def objective(trial):
       x = omlflow.suggest_uniform("x", -10, 10)

       results = []

       for i in range(7):  # simulate 7 fold cross-validation
           result = (x - 2) ** 2
           omlflow.log_iter({"fold_result": result}, i)  # call to log the fold as nested run
           results.append(result)

       result = np.mean(results)
       return result  # auto logging - no explicit call to log_metric needed

   study = optuna.create_study()
   study.optimize(objective, n_trials=100)

.. note::
   Optuna does not support nested runs.
   That is why the results are aggregated into lists when they are stored
   as user attributes at Optuna.

Set MLflow Tracking Server URI
------------------------------

.. todo::
   add content

Enforce no uncommited GIT Changes
---------------------------------

.. todo::
   add content
