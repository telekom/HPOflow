# Copyright (c) 2021 Philip May
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import numpy as np
import optuna
import pytest
from mlflow.tracking import MlflowClient

from hpoflow.optuna_mlflow import OptunaMLflow


def _objective_func_factory(tracking_uri, num_folds):
    @OptunaMLflow(tracking_uri=tracking_uri)
    def _objective_func(omlflow):
        x = omlflow.suggest_uniform("x", -10, 10)
        results = []

        # do folds
        for i in range(num_folds):
            result = (x - 2) ** 2
            omlflow.log_iter({"result": result}, i)
            results.append(result)

        result = np.mean(results)
        return result

    return _objective_func


def test_integration_study_name_run_data_to_file(tmpdir):
    tracking_file_name = "file:{}".format(tmpdir)
    study_name = "my_study"
    n_trials = 2
    num_folds = 3

    study = optuna.create_study(study_name=study_name)
    study.optimize(_objective_func_factory(tracking_file_name, num_folds), n_trials=n_trials)

    mlfl_client = MlflowClient(tracking_file_name)
    experiments = mlfl_client.list_experiments()
    assert len(experiments) == 1

    experiment = experiments[0]
    assert experiment.name == study_name
    experiment_id = experiment.experiment_id

    run_infos = mlfl_client.list_run_infos(experiment_id)
    assert len(run_infos) == n_trials + n_trials * num_folds

    first_run_id = run_infos[-1].run_id
    first_run = mlfl_client.get_run(first_run_id)
    first_run_dict = first_run.to_dictionary()
    assert "x" in first_run_dict["data"]["params"]
    assert first_run_dict["data"]["tags"]["direction"] == "MINIMIZE"


def test_integration_exception_wrong_url(caplog):
    """Test handling of MLflow connection problems."""
    tracking_uri = "http://not.available"
    study_name = "my_study"
    n_trials = 2
    num_folds = 2

    study = optuna.create_study(study_name=study_name)

    with pytest.warns(RuntimeWarning):
        study.optimize(_objective_func_factory(tracking_uri, num_folds), n_trials=n_trials)

    assert len(caplog.records) > 0

    for log_record in caplog.records:
        assert "HTTPConnectionPool" in log_record.getMessage()
