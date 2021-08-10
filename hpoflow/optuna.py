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
    As the test method a t-test is used. Our experiments have shown that an ``aplha`` value
    between 0.3 and 0.4 is reasonable.
    """

    def __init__(self, alpha: float = 0.1, n_warmup_steps: int = 4) -> None:
        """Constructor.

        Args:
            alpha: The alpha level for the statistical significance test.
                The larger this value is, the more aggressively this pruner works.
                The smaller this value is, the stronger the statistical difference between the two
                distributions must be for Optuna to prune.
                ``alpha`` must be ``0 < alpha < 1``.
            n_warmup_steps: Pruning is disabled until the trial reaches or exceeds the given number
                of steps.
        """
        # input value check
        if n_warmup_steps < 0:
            raise ValueError(
                "'n_warmup_steps' must not be negative! n_warmup_steps: {}".format(n_warmup_steps)
            )
        if alpha >= 1:
            raise ValueError("'alpha' must be smaller than 1! {}".format(alpha))
        if alpha <= 0:
            raise ValueError("'alpha' must be greater than 0! {}".format(alpha))

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

            _logger.debug("trial_intermediate_values: %s", trial_intermediate_values)

            # wait until the trial reaches or exceeds n_warmup_steps number of steps
            if len(trial_intermediate_values) >= self.n_warmup_steps:
                trial_mean = np.mean(trial_intermediate_values)

                best_trial_intermediate_values = list(best_trial.intermediate_values.values())
                best_trial_mean = np.mean(best_trial_intermediate_values)

                _logger.debug("trial_mean: %s", trial_mean)
                _logger.debug("best_trial_intermediate_values: %s", best_trial_intermediate_values)
                _logger.debug("best_trial_mean: %s", best_trial_mean)

                if (
                    trial_mean < best_trial_mean and study.direction == StudyDirection.MAXIMIZE
                ) or (trial_mean > best_trial_mean and study.direction == StudyDirection.MINIMIZE):

                    pvalue = stats.ttest_ind(
                        trial_intermediate_values,
                        best_trial_intermediate_values,
                    ).pvalue

                    _logger.debug("pvalue: %s", pvalue)

                    if pvalue < self.alpha:
                        _logger.info("We prune this trial. pvalue: %s", pvalue)

                        return True

                else:
                    _logger.debug(
                        "This trial is better than best trial - we do not check for pruning."
                    )

            else:
                _logger.debug("This trial did not reach n_warmup_steps - we do no checks.")

        return False
