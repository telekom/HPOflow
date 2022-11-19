# Copyright (c) 2021 Philip May
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT
from unittest.mock import patch

import numpy as np
import optuna
import pytest
from mlflow.tracking import MlflowClient

from hpoflow import OptunaMLflow


_trial_exception_text = "This fake trial raised an exception!"
_trial_tag_key = "some-tag-key"
_trial_tag_value = "some-tag-value"
_trial_metric_key = "some-metric-key"
_trial_metric_value = 7.0
_trial_inner_metric_key = "inner-result"


def _objective_func_factory(kwargs, num_folds, raise_exception_type=None):
    @OptunaMLflow(**kwargs)
    def _objective_func(omlflow):
        x = omlflow.suggest_uniform("x", -10, 10)

        if raise_exception_type is not None:
            raise raise_exception_type(_trial_exception_text)

        results = []

        # do folds
        for i in range(num_folds):
            result = (x - 2) ** 2
            omlflow.log_iter({_trial_inner_metric_key: result}, i)
            results.append(result)

        omlflow.set_tag(_trial_tag_key, _trial_tag_value)
        omlflow.log_metric(_trial_metric_key, _trial_metric_value)

        result = np.mean(results)
        return result

    return _objective_func


def test_integration_to_file(tmpdir):
    tracking_file_name = "file:{}".format(tmpdir)
    study_name = "my_study"
    n_trials = 2
    num_folds = 3

    study = optuna.create_study(study_name=study_name)
    study.optimize(
        _objective_func_factory({"tracking_uri": tracking_file_name}, num_folds), n_trials=n_trials
    )

    mlfl_client = MlflowClient(tracking_file_name)
    experiments = mlfl_client.list_experiments()
    assert len(experiments) == 1

    experiment = experiments[0]
    assert experiment.name == study_name
    experiment_id = experiment.experiment_id

    run_infos = mlfl_client.list_run_infos(experiment_id)
    assert len(run_infos) == n_trials + n_trials * num_folds

    # get first (outer) MLflow run
    first_run_id = run_infos[-1].run_id
    first_run = mlfl_client.get_run(first_run_id)
    first_run_dict = first_run.to_dictionary()

    # test info of first (outer) MLflow run
    assert first_run_dict["info"]["status"] == "FINISHED"

    # test data.metrics of first (outer) MLflow run
    assert "optuna_result" in first_run_dict["data"]["metrics"]
    assert first_run_dict["data"]["metrics"][_trial_metric_key] == _trial_metric_value

    # test data.params of first (outer) MLflow run
    assert "x" in first_run_dict["data"]["params"]

    # test data.tags of first (outer) MLflow run
    assert "x_distribution" in first_run_dict["data"]["tags"]
    assert first_run_dict["data"]["tags"]["direction"] == "MINIMIZE"
    assert "process_id" in first_run_dict["data"]["tags"]
    assert first_run_dict["data"]["tags"]["hostname"] != "unknown"
    assert first_run_dict["data"]["tags"]["mlflow.runName"] == "000"
    assert "mlflow.user" in first_run_dict["data"]["tags"]
    assert first_run_dict["data"]["tags"][_trial_tag_key] == _trial_tag_value

    # get first nested MLflow run
    first_nested_run_id = run_infos[-2].run_id
    first_nested_run = mlfl_client.get_run(first_nested_run_id)
    first_nested_run_dict = first_nested_run.to_dictionary()

    # test first nested MLflow run
    assert _trial_inner_metric_key in first_nested_run_dict["data"]["metrics"]
    assert first_nested_run_dict["info"]["status"] == "FINISHED"
    assert first_nested_run_dict["data"]["tags"]["mlflow.runName"] == "000-000"

    # get optuna trials
    trials = study.get_trials()

    # test optuna trials
    assert len(trials) == n_trials

    # get first optuna trial
    first_trial = trials[0]

    # test first optuna trial
    assert "x" in first_trial.params
    assert first_trial.state == optuna.trial.TrialState.COMPLETE
    assert "hostname" in first_trial.user_attrs
    assert "process_id" in first_trial.user_attrs
    assert f"{_trial_inner_metric_key}_iter" in first_trial.user_attrs
    assert first_trial.user_attrs[_trial_tag_key] == _trial_tag_value
    assert first_trial.user_attrs[_trial_metric_key] == _trial_metric_value


def test_set_optuna_result_name(tmpdir):
    tracking_file_name = "file:{}".format(tmpdir)
    study_name = "my_study"
    optuna_result_name = "other_name"
    n_trials = 2
    num_folds = 3

    study = optuna.create_study(study_name=study_name)
    study.optimize(
        _objective_func_factory(
            {"tracking_uri": tracking_file_name, "optuna_result_name": optuna_result_name},
            num_folds,
        ),
        n_trials=n_trials,
    )

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

    assert "optuna_result" not in first_run_dict["data"]["metrics"]
    assert optuna_result_name in first_run_dict["data"]["metrics"]


def test_set_num_name_digits(tmpdir):
    tracking_file_name = "file:{}".format(tmpdir)
    study_name = "my_study"
    num_name_digits = 7
    n_trials = 2
    num_folds = 3

    study = optuna.create_study(study_name=study_name)
    study.optimize(
        _objective_func_factory(
            {"tracking_uri": tracking_file_name, "num_name_digits": num_name_digits},
            num_folds,
        ),
        n_trials=n_trials,
    )

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

    assert first_run_dict["data"]["tags"]["mlflow.runName"] == ("0" * num_name_digits)


@pytest.mark.skip(reason="Loading the wrong URL takes an infinite amount of time.")
def test_integration_exception_wrong_url(caplog):
    """Test handling of MLflow connection problems."""
    tracking_uri = "http://not.available"
    study_name = "my_study"
    n_trials = 2
    num_folds = 2

    study = optuna.create_study(study_name=study_name)

    with pytest.warns(RuntimeWarning):
        study.optimize(
            _objective_func_factory({"tracking_uri": tracking_uri}, num_folds), n_trials=n_trials
        )

    assert len(caplog.records) > 0

    for log_record in caplog.records:
        assert "HTTPConnectionPool" in log_record.getMessage()


def test_failing_trial(tmpdir):
    tracking_file_name = "file:{}".format(tmpdir)
    study_name = "my_study"
    n_trials = 2
    num_folds = 3

    study = optuna.create_study(study_name=study_name)

    with pytest.raises(Exception):
        study.optimize(
            _objective_func_factory(
                {"tracking_uri": tracking_file_name}, num_folds, raise_exception_type=Exception
            ),
            n_trials=n_trials,
        )

    mlfl_client = MlflowClient(tracking_file_name)
    experiments = mlfl_client.list_experiments()
    assert len(experiments) == 1

    experiment = experiments[0]
    assert experiment.name == study_name
    experiment_id = experiment.experiment_id

    run_infos = mlfl_client.list_run_infos(experiment_id)
    assert len(run_infos) == 1

    first_run_id = run_infos[-1].run_id
    first_run = mlfl_client.get_run(first_run_id)
    first_run_dict = first_run.to_dictionary()

    assert first_run_dict["info"]["status"] == "FAILED"
    assert _trial_exception_text in first_run_dict["data"]["tags"]["exception"]
    assert first_run_dict["data"]["metrics"] == {}

    # get optuna trials
    trials = study.get_trials()

    # test optuna trials
    assert len(trials) == 1

    # get first optuna trial
    first_trial = trials[0]

    # test first optuna trial
    assert "x" in first_trial.params
    assert first_trial.state == optuna.trial.TrialState.FAIL
    assert "hostname" in first_trial.user_attrs
    assert "process_id" in first_trial.user_attrs
    assert "exception" in first_trial.user_attrs


def test_killed_trial(tmpdir):
    tracking_file_name = "file:{}".format(tmpdir)
    study_name = "my_study"
    n_trials = 2
    num_folds = 3

    study = optuna.create_study(study_name=study_name)

    with pytest.raises(KeyboardInterrupt):
        study.optimize(
            _objective_func_factory(
                {"tracking_uri": tracking_file_name},
                num_folds,
                raise_exception_type=KeyboardInterrupt,
            ),
            n_trials=n_trials,
        )

    mlfl_client = MlflowClient(tracking_file_name)
    experiments = mlfl_client.list_experiments()
    assert len(experiments) == 1

    experiment = experiments[0]
    assert experiment.name == study_name
    experiment_id = experiment.experiment_id

    run_infos = mlfl_client.list_run_infos(experiment_id)
    assert len(run_infos) == 1

    first_run_id = run_infos[-1].run_id
    first_run = mlfl_client.get_run(first_run_id)
    first_run_dict = first_run.to_dictionary()

    assert first_run_dict["info"]["status"] == "KILLED"
    assert _trial_exception_text in first_run_dict["data"]["tags"]["exception"]
    assert first_run_dict["data"]["metrics"] == {}

    # get optuna trials
    trials = study.get_trials()

    # test optuna trials
    assert len(trials) == 1

    # get first optuna trial
    first_trial = trials[0]

    # test first optuna trial
    assert "x" in first_trial.params
    assert first_trial.state != optuna.trial.TrialState.COMPLETE
    assert "hostname" in first_trial.user_attrs
    assert "process_id" in first_trial.user_attrs
    assert "exception" in first_trial.user_attrs


@patch("hpoflow.optuna_mlflow.check_repo_is_dirty")
def test_enforce_clean_git_not_clean(check_repo_is_dirty_mock, tmpdir):
    exception_text = "Git is dirty mock error!"
    check_repo_is_dirty_mock.side_effect = RuntimeError(exception_text)
    tracking_file_name = "file:{}".format(tmpdir)
    study_name = "my_study"
    n_trials = 2
    num_folds = 3

    study = optuna.create_study(study_name=study_name)

    with pytest.raises(RuntimeError) as excinfo:
        study.optimize(
            _objective_func_factory(
                {"tracking_uri": tracking_file_name, "enforce_clean_git": True}, num_folds
            ),
            n_trials=n_trials,
        )

    check_repo_is_dirty_mock.assert_called_with()
    assert exception_text in str(excinfo.value)
