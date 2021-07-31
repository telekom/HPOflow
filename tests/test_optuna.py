# Copyright (c) 2021 Philip May
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

import pytest

from hpoflow import SignificanceRepeatedTrainingPruner


def test_percentile_pruner_n_warmup_steps() -> None:
    SignificanceRepeatedTrainingPruner(n_warmup_steps=2)
    SignificanceRepeatedTrainingPruner(n_warmup_steps=0)

    with pytest.raises(ValueError):
        SignificanceRepeatedTrainingPruner(n_warmup_steps=-1)


def test_percentile_pruner_alpha() -> None:
    SignificanceRepeatedTrainingPruner(alpha=0.5)

    with pytest.raises(ValueError):
        SignificanceRepeatedTrainingPruner(alpha=0)

    with pytest.raises(ValueError):
        SignificanceRepeatedTrainingPruner(alpha=1)

    with pytest.raises(ValueError):
        SignificanceRepeatedTrainingPruner(alpha=-0.1)

    with pytest.raises(ValueError):
        SignificanceRepeatedTrainingPruner(alpha=1.1)
