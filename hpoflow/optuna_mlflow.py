# Copyright (c) 2021 Philip May
# Copyright (c) 2021 Philip May, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Wrapper to log to Optuna and MLflow at the same time."""

import logging
import os
import platform
import sys
import textwrap
import traceback
import warnings
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import mlflow
import optuna
from mlflow.entities import RunStatus
from optuna.distributions import CategoricalChoiceType

from hpoflow.mlflow import (
    check_repo_is_dirty,
    normalize_mlflow_entry_name,
    normalize_mlflow_entry_names_in_dict,
)
from hpoflow.utils import func_no_exception_caller


_logger = logging.getLogger(__name__)
_max_mlflow_tag_length = 5000


class OptunaMLflow:
    """Wrapper to log to Optuna and MLflow at the same time."""

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        num_name_digits: int = 3,
        enforce_clean_git: bool = False,
        optuna_result_name: str = "optuna_result",
    ):
        """Constructor.

        Args:
            tracking_uri: The MLflow tracking URL. Defaults to ``None`` which logs to the default
                locale folder ``./mlruns`` or uses the ``MLFLOW_TRACKING_URI`` environment variable
                if it is available. Also see :func:`mlflow.set_tracking_uri`.
            num_name_digits: Number of digits for the MLflow ``run_name``.
            enforce_clean_git: Check and enforce that the GIT repository has no uncommited changes
                (see :meth:`git.repo.base.Repo.is_dirty`).
            optuna_result_name: Name of the metric which is logged to MLflo and is returned by the
                objective function.
        """
        # TODO: add checks for num_name_digits and optuna_result_name

        self._tracking_uri = tracking_uri
        self._num_name_digits = num_name_digits
        self._enforce_clean_git = enforce_clean_git
        self._optuna_result_name = optuna_result_name

        self._hostname: Optional[str] = None

    def __call__(
        self,
        # we use a strange type annotation here
        # see https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class  # noqa: E501
        func: Callable[[Union[optuna.trial.Trial, "OptunaMLflow"]], float],
    ) -> Callable[[optuna.trial.Trial], float]:
        """Returns the decorator for the Optuna objective function.

        Args:
            func: The optuna objective function for the decorator.
        """

        @wraps(func)
        def objective_decorator(trial: optuna.trial.Trial) -> float:
            """Decorator for the Optuna objective function."""
            # we must do this here and not in __init__
            # __init__ is only called once when decorator is applied
            # pylint: disable=attribute-defined-outside-init
            self._trial = trial
            self._iter_metrics: Dict[str, List[float]] = {}
            self._next_iter_num: int = 0

            # check if GIT repo is clean
            if self._enforce_clean_git:
                check_repo_is_dirty()
                # TODO: set a tag if it is clean or when not checked

            try:
                # set tracking_uri for MLflow
                if self._tracking_uri is not None:
                    mlflow.set_tracking_uri(self._tracking_uri)

                mlflow.set_experiment(self._trial.study.study_name)

                digits_format_string = "{{:0{}d}}".format(self._num_name_digits)
                mlflow.start_run(run_name=digits_format_string.format(self._trial.number))
            except Exception as e:
                error_msg = "Exception raised during MLflow communication! Exception: {}".format(e)
                _logger.error(error_msg, exc_info=True)
                warnings.warn(error_msg, RuntimeWarning)

            _logger.info("Run %s started.", self._trial.number)

            tag_dict = {
                "hostname": self._get_hostname(),
                "process_id": os.getpid(),
            }
            self.set_tags(tag_dict)

            try:
                # call objective function
                result = func(self)

                # log the result to MLflow but not optuna
                self.log_metric(self._optuna_result_name, result, optuna_log=False)

                # extract and set tags from trial
                tags = {}
                # Set direction and convert it to str and remove the common prefix.
                study_direction = self._trial.study.direction
                if isinstance(study_direction, optuna._study_direction.StudyDirection):
                    tags["direction"] = str(study_direction).rsplit(".", maxsplit=1)[-1]

                distributions = {
                    (k + "_distribution"): str(v) for (k, v) in self._trial.distributions.items()
                }
                tags.update(distributions)
                self.set_tags(tags, optuna_log=False)

                # end run
                self._end_run(RunStatus.to_string(RunStatus.FINISHED))
                _logger.info("Run finished.")

                return result
            except (Exception, KeyboardInterrupt) as e:
                error_msg = "Exception raised while executing Optuna trial! Exception: {}".format(
                    e
                )
                _logger.error(error_msg, exc_info=True)

                # log exception info to Optuna and MLflow as a tag
                exc_type, exc_value, exc_traceback = sys.exc_info()
                exc_text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                self.set_tag("exception", exc_text)
                if exc_type is KeyboardInterrupt:
                    self._end_run(RunStatus.to_string(RunStatus.KILLED))
                    _logger.info("Run killed.")
                else:
                    self._end_run(RunStatus.to_string(RunStatus.FAILED))
                    _logger.info("Run failed.")
                    warnings.warn(error_msg, RuntimeWarning)
                raise  # raise exception again

        return objective_decorator

    #####################################
    # MLflow wrapper functions
    #####################################

    def log_metric(
        self, key: str, value: float, step: Optional[int] = None, optuna_log: Optional[bool] = True
    ) -> None:
        """Log a metric under the current run.

        Wrapper of the corresponding MLflow function (see :func:`mlflow.log_metric`). The data is
        logged to MLflow and also added to Optuna as a user attribute (see
        :meth:`optuna.trial.Trial.set_user_attr`).

        Args:
            key: x
            value: x
            step: x
            optuna_log: If ``False`` this is not logged to Optuna. This is an internal parameter
                that should be ignored by the API user.
        """
        if optuna_log:
            self._trial.set_user_attr(key, value)
        _logger.info("Metric: %s: %s at step: %s", key, value, step)
        func_no_exception_caller(
            mlflow.log_metric, normalize_mlflow_entry_name(key), value, step=None
        )

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        optuna_log: Optional[bool] = True,
    ) -> None:
        """Log multiple metrics for the current run.

        Wrapper of the corresponding MLflow function (see :func:`mlflow.log_metrics`). The data is
        logged to MLflow and also added to Optuna as a user attribute (see
        :meth:`optuna.trial.Trial.set_user_attr`).

        Args:
            metrics: x
            step: x
            optuna_log: If ``False`` this is not logged to Optuna. This is an internal parameter
                that should be ignored by the API user.
        """
        for key, value in metrics.items():
            if optuna_log:
                self._trial.set_user_attr(key, value)
            _logger.info("Metric: %s: %s at step: %s", key, value, step)
        func_no_exception_caller(
            mlflow.log_metrics, normalize_mlflow_entry_names_in_dict(metrics), step=step
        )

    def log_param(self, key: str, value: Any, optuna_log: Optional[bool] = True) -> None:
        """Log a parameter under the current run.

        Wrapper of the corresponding MLflow function (see :func:`mlflow.log_param`). The data is
        logged to MLflow and also added to Optuna as a user attribute (see
        :meth:`optuna.trial.Trial.set_user_attr`).

        Args:
            key: x
            value: x
            optuna_log: If ``False`` this is not logged to Optuna. This is an internal parameter
                that should be ignored by the API user.
        """
        if optuna_log:
            self._trial.set_user_attr(key, value)
        _logger.info("Param: %s: %s", key, value)
        func_no_exception_caller(mlflow.log_param, normalize_mlflow_entry_name(key), value)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log a batch of params for the current run.

        Wrapper of the corresponding MLflow function (see :func:`mlflow.log_params`). The data is
        logged to MLflow and also added to Optuna as a user attribute (see
        :meth:`optuna.trial.Trial.set_user_attr`).
        """
        for key, value in params.items():
            self._trial.set_user_attr(key, value)
            _logger.info("Param: %s: %s", key, value)
        func_no_exception_caller(mlflow.log_params, normalize_mlflow_entry_names_in_dict(params))

    def set_tag(self, key: str, value: Any, optuna_log: Optional[bool] = True) -> None:
        """Set a tag under the current run.

        Wrapper of the corresponding MLflow function (see :func:`mlflow.set_tag`). The data is
        logged to MLflow and also added to Optuna as a user attribute (see
        :meth:`optuna.trial.Trial.set_user_attr`).

        Args:
            key: x
            value: x
            optuna_log: If ``False`` this is not logged to Optuna. This is an internal parameter
                that should be ignored by the API user.
        """
        if optuna_log:
            self._trial.set_user_attr(key, value)
        _logger.info("Tag: %s: %s", key, value)
        value = str(value)  # make sure it is a string
        if len(value) > _max_mlflow_tag_length:
            value = textwrap.shorten(value, _max_mlflow_tag_length)
        func_no_exception_caller(mlflow.set_tag, normalize_mlflow_entry_name(key), value)

    def set_tags(self, tags: Dict[str, Any], optuna_log: Optional[bool] = True) -> None:
        """Log a batch of tags for the current run.

        Wrapper of the corresponding MLflow function (see :func:`mlflow.set_tags`). The data is
        logged to MLflow and also added to Optuna as a user attribute (see
        :meth:`optuna.trial.Trial.set_user_attr`).

        Args:
            tags: x
            optuna_log: If ``False`` this is not logged to Optuna. This is an internal parameter
                that should be ignored by the API user.
        """
        for key, value in tags.items():
            if optuna_log:
                self._trial.set_user_attr(key, value)
            _logger.info("Tag: %s: %s", key, value)
            value = str(value)  # make sure it is a string
            if len(value) > _max_mlflow_tag_length:
                tags[key] = textwrap.shorten(value, _max_mlflow_tag_length)
        func_no_exception_caller(mlflow.set_tags, normalize_mlflow_entry_names_in_dict(tags))

    def log_iter(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log an iteration or a fold as a nested run (see :func:`mlflow.log_metrics`).

        The data is logged to MLflow and also added to Optuna as a user attribute (see
        :meth:`optuna.trial.Trial.set_user_attr`).
        """
        for key, value in metrics.items():
            value_list: List[float] = self._iter_metrics.get(key, [])
            value_list.append(value)
            self._iter_metrics[key] = value_list
            self._trial.set_user_attr("{}_iter".format(key), value_list)
            _logger.info("Iteration metric: %s: %s at step: %s", key, value, step)
        digits_format_string = "{{:0{0}d}}-{{:0{0}d}}".format(self._num_name_digits)
        if step is None:
            step = self._next_iter_num
            self._next_iter_num += 1
        func_no_exception_caller(
            self._log_iter,
            run_name=digits_format_string.format(self._trial.number, step),
            metrics=metrics,
            step=step,
        )

    def _log_iter(self, run_name: str, metrics: Dict[str, float], step: int):
        """Log an iteration or a fold as a nested run (see :func:`mlflow.log_metrics`).

        The data is logged only to MLflow and not to Optuna.
        """
        with mlflow.start_run(run_name=run_name, nested=True):
            self.log_metrics(metrics, step=step, optuna_log=False)

    @staticmethod
    def _end_run(status: str, exc_text=None) -> None:
        """End the active MLflow run (see :func:`mlflow.end_run`).

        Args:
            status: The status of the run (see :class:`mlflow.entities.RunStatus`).
            exc_text: x
        """
        func_no_exception_caller(mlflow.end_run, status)
        if exc_text is None:
            _logger.info("Run finished with status: %s", status)
        else:
            _logger.error("Run finished with status: %s, exc_text: %s", status, exc_text)

    #####################################
    # util functions
    #####################################

    def _get_hostname(self) -> str:
        """Get the hostname."""
        if self._hostname is None:
            self._hostname = "unknown"
            try:
                self._hostname = platform.node()
            except Exception as e:
                warn_msg = "Exception while getting hostname! {}".format(e)
                _logger.warning(warn_msg)
                warnings.warn(warn_msg, RuntimeWarning)
        return self._hostname

    #####################################
    # Optuna wrapper functions
    #####################################

    def report(self, value: float, step: int) -> None:
        """Report an objective function value for a given step.

        Wrapper of the corresponding Optuna function (see :meth:`optuna.trial.Trial.report`).

        Args:
            value: A value returned from the evaluation.
            step: Step of the trial (e.g., Epoch of neural network training). Note that pruners
                assume that ``step`` starts at zero. For example,
        """
        self._trial.report(value, step)

    def should_prune(self) -> bool:
        """Suggest whether the trial should be pruned or not.

        Wrapper of the corresponding Optuna function (see :meth:`optuna.trial.Trial.should_prune`).
        """
        return self._trial.should_prune()

    def suggest_categorical(
        self, name: str, choices: Sequence[CategoricalChoiceType]
    ) -> CategoricalChoiceType:
        """Suggest a value for the categorical parameter.

        Wrapper of the corresponding Optuna function (see
        :meth:`optuna.trial.Trial.suggest_categorical`).

        Args:
            name: A parameter name.
            choices: Parameter value candidates.
        """
        result = self._trial.suggest_categorical(name, choices)
        self.log_param(name, result, optuna_log=False)
        return result

    def suggest_discrete_uniform(self, name: str, low: float, high: float, q: float) -> float:
        """Suggest a value for the discrete parameter.

        Wrapper of the corresponding Optuna function (see
        :meth:`optuna.trial.Trial.suggest_discrete_uniform`).

        Args:
            name: A parameter name.
            low: Lower endpoint of the range of suggested values. ``low`` is included in the range.
            high: Upper endpoint of the range of suggested values. ``high`` is included in the
                range.
            q: A step of discretization.
        """
        result = self._trial.suggest_discrete_uniform(name, low, high, q)
        self.log_param(name, result, optuna_log=False)
        return result

    def suggest_int(self, name: str, low: int, high: int, step: int = 1, log: bool = False) -> int:
        """Suggest a value for the integer parameter.

        Wrapper of the corresponding Optuna function (see :meth:`optuna.trial.Trial.suggest_int`).

        Args:
            name: A parameter name.
            low: Lower endpoint of the range of suggested values. ``low`` is included in the range.
            high: Upper endpoint of the range of suggested values. ``high`` is included in the
                range.
            step: A step of discretization.
            log: A flag to sample the value from the log domain or not.
        """
        result = self._trial.suggest_int(name, low, high, step, log)
        self.log_param(name, result, optuna_log=False)
        return result

    def suggest_loguniform(self, name: str, low: float, high: float) -> float:
        """Suggest a value in the log domain for the continuous parameter.

        Wrapper of the corresponding Optuna function (see
        :meth:`optuna.trial.Trial.suggest_loguniform`).

        Args:
            name: A parameter name.
            low: Lower endpoint of the range of suggested values. ``low`` is included in the range.
            high: Upper endpoint of the range of suggested values. ``high`` is excluded from the
                range.
        """
        result = self._trial.suggest_loguniform(name, low, high)
        self.log_param(name, result, optuna_log=False)
        return result

    def suggest_uniform(self, name: str, low: float, high: float) -> float:
        """Suggest a value for the continuous parameter.

        Wrapper of the corresponding Optuna function (see
        :meth:`optuna.trial.Trial.suggest_uniform`).

        Args:
            name: A parameter name.
            low: Lower endpoint of the range of suggested values. ``low`` is included in the range.
            high: Upper endpoint of the range of suggested values. ``high`` is excluded from the
                range.
        """
        result = self._trial.suggest_uniform(name, low, high)
        self.log_param(name, result, optuna_log=False)
        return result
