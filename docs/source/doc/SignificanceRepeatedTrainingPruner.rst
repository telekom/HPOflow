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

.. code-block:: python

    from hpoflow import SignificanceRepeatedTrainingPruner
    import logging
    import numpy as np
    import optuna
    from sklearn.datasets import load_iris
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    # configure the logger to see the debug output from the pruner
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger("hpoflow.optuna").setLevel(logging.DEBUG)

    dataset = load_iris()

    x, y = dataset['data'], dataset['target']

    def train(trial):
        parameter = {
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'n_estimators': trial.suggest_int('n_estimators', 20, 100),
        }

        validation_result_list = []

        skf = StratifiedKFold(n_splits=10)
        for fold_index, (train_index, val_index) in enumerate(skf.split(x, y)):
            X_train, X_val = x[train_index], x[val_index]
            y_train, y_val = y[train_index], y[val_index]

            rf = RandomForestClassifier(**parameter)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_val)

            acc = accuracy_score(y_val, y_pred)
            validation_result_list.append(acc)

            # report result of this fold
            trial.report(acc, fold_index)

            # check if we should prune
            if trial.should_prune():
                # prune here - we are done with this CV
                break

        return np.mean(validation_result_list)

    study = optuna.create_study(
        storage="sqlite:///optuna.db",
        study_name="iris_cv",
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(multivariate=True),
        # add pruner to optuna
        pruner=SignificanceRepeatedTrainingPruner(
            alpha=0.4,
            n_warmup_steps=4,
        )
    )

    study.optimize(train, n_trials=10)

.. todo::
   - add more details
   - improve example
