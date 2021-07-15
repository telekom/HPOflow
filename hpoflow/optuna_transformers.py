# Copyright (c) 2021 Timothy Wolff-Piggott
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Integration of Optuna and Transformers."""

import logging
import os
from numbers import Number
from typing import Dict, Union

import mlflow
import transformers
from transformers import TrainerControl, TrainerState, TrainingArguments

from hpoflow.optuna_mlflow import OptunaMLflow


_logger = logging.getLogger(__name__)


class OptunaMLflowCallback(transformers.TrainerCallback):
    """Integration of Optuna and Transformers.

    Class based on :class:`transformers.TrainerCallback`; integrates with OptunaMLflow to send
    the logs to ``MLflow`` and ``Optuna`` during model training.
    """

    def __init__(
        self,
        trial: OptunaMLflow,
        log_training_args: bool = True,
        log_model_config: bool = True,
    ):
        """Constructor.

        Args:
            trial: The OptunaMLflow object.
            log_training_args: Whether to log all Transformers TrainingArguments as MLflow params.
            log_model_config: Whether to log the Transformers model config as MLflow params.
        """
        self._trial = trial
        self._log_training_args = log_training_args
        self._log_model_config = log_model_config

        self._initialized = False
        self._log_artifacts = False

    def setup(
        self,
        args: TrainingArguments,
        state: TrainerState,
        model: Union[transformers.PreTrainedModel, transformers.TFPreTrainedModel],
    ):
        """Setup the optional MLflow integration.

        You can set the environment variable ``HF_MLFLOW_LOG_ARTIFACTS``. It is to use
        :func:`mlflow.log_artifacts` to log artifacts. This only makes sense if logging to a remote
        server, e.g. s3 or GCS. If set to ``True`` or ``1``, will copy whatever is in
        TrainerArgument's output_dir to the local or remote artifact storage. Using it without a
        remote storage will just copy the files to your artifact location.
        """
        log_artifacts = os.getenv("HF_MLFLOW_LOG_ARTIFACTS", "FALSE").upper()
        if log_artifacts in {"TRUE", "1", "YES"}:
            self._log_artifacts = True
        if state.is_world_process_zero:
            combined_dict = dict()
            if self._log_training_args:
                _logger.info("Logging training arguments.")
                combined_dict.update(args.to_dict())
            if self._log_model_config and hasattr(model, "config") and model.config is not None:
                _logger.info("Logging model config.")
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            # remove params that are too long for MLflow
            for name, value in list(combined_dict.items()):
                # internally, all values are converted to str in MLflow
                if len(str(value)) > mlflow.utils.validation.MAX_PARAM_VAL_LENGTH:
                    _logger.warning(
                        "Trainer is attempting to log a value of "
                        "'%s' for key '%s' as a parameter. "
                        "MLflow's log_param() only accepts values no longer than "
                        "250 characters so we dropped this attribute.",
                        value,
                        name,
                    )
                    del combined_dict[name]
            # MLflow cannot log more than 100 values in one go, so we have to split it
            combined_dict_items = list(combined_dict.items())
            for i in range(
                0, len(combined_dict_items), mlflow.utils.validation.MAX_PARAMS_TAGS_PER_BATCH
            ):
                self._trial.log_params(
                    dict(
                        combined_dict_items[
                            i : i + mlflow.utils.validation.MAX_PARAMS_TAGS_PER_BATCH
                        ]
                    )
                )
        self._initialized = True

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Union[transformers.PreTrainedModel, transformers.TFPreTrainedModel] = None,
        **kwargs,
    ) -> None:
        """Event called at the beginning of training.

        Call setup if not yet initialized.
        """
        if not self._initialized:
            self.setup(args, state, model)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, Number],
        model: Union[transformers.PreTrainedModel, transformers.TFPreTrainedModel] = None,
        **kwargs,
    ):
        """Event called after logging the last logs.

        Log all metrics from Transformers logs as MLflow metrics at the appropriate step.
        """
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            metrics_to_log: Dict[str, float] = dict()
            for k, v in logs.items():
                if isinstance(v, (int, float)):  # TODO: remove or change to Number?
                    metrics_to_log[k] = v
                else:
                    _logger.warning(
                        "Trainer is attempting to log a value of "
                        "'%s' of type %s for key '%s' as a metric. "
                        "MLflow's log_metric() only accepts float and "
                        "int types so we dropped this attribute.",
                        v,
                        type(v),
                        k,
                    )
            self._trial.log_metrics(metrics_to_log, step=state.global_step)

    def on_train_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ):
        """Event called at the end of training.

        Log the training output as MLflow artifacts if logging artifacts is enabled.
        """
        if self._initialized and state.is_world_process_zero:
            if self._log_artifacts:
                _logger.info("Logging artifacts. This may take time.")
                mlflow.log_artifacts(args.output_dir)
