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

By passing ``tracking_uri`` to the constructor of :class:`~hpoflow.optuna_mlflow.OptunaMLflow`
you can set the MLflow tracking server URI (also see :func:`mlflow.set_tracking_uri`).
The values can be:

- not set or ``None``: MLflow logs to the default
  locale folder ``./mlruns`` or uses the ``MLFLOW_TRACKING_URI`` environment variable
  if it is available.
- local file path, prefixed with ``file:/``:
  Data is stored locally at the provided directory.
- HTTP URI like ``https://my-tracking-server:5000``
- Databricks workspace, provided as the string ``databricks`` or, to use a
  `Databricks CLI profile <https://github.com/databricks/databricks-cli>`__,
  ``databricks://<profileName>``

Enforce no uncommitted GIT Changes
---------------------------------

By passing ``enforce_clean_git=True`` to the constructor of
:class:`~hpoflow.optuna_mlflow.OptunaMLflow` you can check and enforce that the
GIT repository has no uncommitted changes (see :meth:`git.repo.base.Repo.is_dirty`).
If there are uncommitted GIT changes an exception is raised.
In this way, reproducibility of experiments is facilitated.
