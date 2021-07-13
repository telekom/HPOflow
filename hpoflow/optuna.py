# Copyright (c) 2021 Philip May
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Optuna only functionality."""

import logging

import numpy as np
import optuna
from optuna.pruners import BasePruner
from optuna.study import StudyDirection
from scipy import stats


_logger = logging.getLogger(__name__)


class SignificanceRepeatedTrainingPruner(BasePruner):
    """Pruner which uses statistical significance as an heuristic for decision-making.

    Pruner to use statistical significance to prune repeated trainings like in a cross validation.
    As the test method a t-test is used.
    """

    def __init__(self, alpha=0.1, n_warmup_steps=4) -> None:
        """Constructor.

        Args:
            alpha: The alpha level for the statistical significance test.
            n_warmup_steps: Pruning is disabled until the trial reaches or exceeds the given number
                of steps.
        """
        if n_warmup_steps < 0:
            raise ValueError(
                "Number of warmup steps cannot be negative but got {}.".format(n_warmup_steps)
            )
        # TODO: check alpha for valid value range and add test

        self.n_warmup_steps = n_warmup_steps
        self.alpha = alpha

    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
        """Judge whether the trial should be pruned based on the reported values."""
        # get best tial - best trial is not available for first trial
        best_trial = None
        try:
            best_trial = study.best_trial
        except ValueError:
            pass

        if best_trial is not None:
            trial_intermediate_values = list(trial.intermediate_values.values())

            # TODO: remove logging or change to debug level
            _logger.info("### SignificanceRepeatedTrainingPruner ###")
            _logger.info("trial_intermediate_values: %s", trial_intermediate_values)

            # wait until the trial reaches or exceeds n_warmup_steps number of steps
            if len(trial_intermediate_values) >= self.n_warmup_steps:
                trial_mean = np.mean(trial_intermediate_values)

                best_trial_intermediate_values = list(best_trial.intermediate_values.values())
                best_trial_mean = np.mean(best_trial_intermediate_values)

                # TODO: remove logging or change to debug level
                _logger.info("trial_mean: %s", trial_mean)
                _logger.info("best_trial_intermediate_values: %s", best_trial_intermediate_values)
                _logger.info("best_trial_mean: %s", best_trial_mean)

                if (
                    trial_mean < best_trial_mean and study.direction == StudyDirection.MAXIMIZE
                ) or (trial_mean > best_trial_mean and study.direction == StudyDirection.MINIMIZE):
                    if study.direction == StudyDirection.MAXIMIZE:
                        alternative = "less"
                    elif study.direction == StudyDirection.MINIMIZE:
                        alternative = "greater"
                    else:
                        raise RuntimeError("Can not find valid StudyDirection!")

                    pvalue = stats.ttest_ind(
                        trial_intermediate_values,
                        best_trial_intermediate_values,
                        alternative=alternative,
                    ).pvalue

                    # TODO: remove logging or change to debug level
                    _logger.info("pvalue: %s", pvalue)

                    if pvalue < self.alpha:
                        # TODO: remove logging or change to debug level
                        _logger.info("We prune this.")

                        return True

                # TODO: remove logging or change to debug level
                else:
                    _logger.info(
                        "This trial is better than best trial - we do not check for pruning."
                    )

            # TODO: remove logging or change to debug level
            else:
                _logger.info("This trial did not reach n_warmup_steps - we do no checks.")

        return False
