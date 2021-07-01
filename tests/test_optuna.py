# Copyright (c) 2021 Philip May
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import pytest

from hpoflow.optuna import SignificanceRepeatedTrainingPruner


def test_percentile_pruner_n_warmup_steps() -> None:
    SignificanceRepeatedTrainingPruner(0.2, n_warmup_steps=2)
    SignificanceRepeatedTrainingPruner(0.2, n_warmup_steps=4)

    with pytest.raises(ValueError):
        SignificanceRepeatedTrainingPruner(0.2, n_warmup_steps=-1)
