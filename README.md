# HPOflow - [Sphinx DOC](https://telekom.github.io/HPOflow/)

[![DOC](https://img.shields.io/badge/DOC-Sphinx-blue)](https://telekom.github.io/HPOflow/)
[![MIT License](https://img.shields.io/github/license/telekom/HPOflow)](https://github.com/telekom/HPOflow/blob/main/LICENSE)
[![Contributor Covenant v2.0](https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg)](https://github.com/telekom/HPOflow/blob/main/CODE_OF_CONDUCT.md)
[![Python Version](https://img.shields.io/pypi/pyversions/hpoflow)](https://www.python.org)
[![pypi](https://img.shields.io/pypi/v/hpoflow.svg)](https://pypi.python.org/pypi/hpoflow)
<br/>
[![pytest status](https://github.com/telekom/HPOflow/actions/workflows/pytest.yml/badge.svg)](https://github.com/telekom/HPOflow/actions/workflows/pytest.yml)
[![Static Code Checks status](https://github.com/telekom/HPOflow/actions/workflows/static_checks.yml/badge.svg)](https://github.com/telekom/HPOflow/actions/workflows/static_checks.yml)
[![Build & Deploy Doc](https://github.com/telekom/HPOflow/actions/workflows/build_deploy_doc.yml/badge.svg)](https://github.com/telekom/HPOflow/actions/workflows/build_deploy_doc.yml)
[![GitHub issues](https://img.shields.io/github/issues-raw/telekom/HPOflow)](https://github.com/telekom/HPOflow/issues)

Tools for [Optuna](https://optuna.readthedocs.io/), [MLflow](https://www.mlflow.org/docs/latest/index.html) and the integration of both.

[![One Conversation](https://raw.githubusercontent.com/telekom/HPOflow/main/docs/source/imgs/1c-logo.png)](https://www.welove.ai/)
<br/>
This project is maintained by the [One Conversation](https://www.welove.ai/)
team of [Deutsche Telekom AG](https://www.telekom.com/).

The main components are:

- [`hpoflow.OptunaMLflow`](https://github.com/telekom/HPOflow/blob/main/hpoflow/optuna_mlflow.py):<br/>
  A wrapper to use Optuna and log to MLflow at the same time.
- [`hpoflow.OptunaMLflowCallback`](https://github.com/telekom/HPOflow/blob/main/hpoflow/optuna_transformers.py):<br/>
  Class inheriting from `transformers.TrainerCallback` that integrates with `OptunaMLflow`
  to send the logs to MLflow and Optuna during model training.
- [`hpoflow.SignificanceRepeatedTrainingPruner`](https://github.com/telekom/HPOflow/blob/main/hpoflow/optuna.py):<br/>
  An [Optuna pruner](https://optuna.readthedocs.io/en/stable/reference/pruners.html)
  to use statistical significance (a t-test which serves as a heuristic) to stop
  unpromising trials early, avoiding unnecessary repeated training during cross validation.

## Installation

HPOflow is available at [the Python Package Index (PyPI)](https://pypi.org/project/hpoflow/).
It can be installed with _pip_:

```bash
$ pip install hpoflow
```

Some additional dependencies might be necessary.

To use [`hpoflow.optuna_mlflow.OptunaMLflow`](https://github.com/telekom/HPOflow/blob/main/hpoflow/optuna_mlflow.py):

```bash
$ pip install mlflow GitPython
```

To use [`hpoflow.optuna_transformers.OptunaMLflowCallback`](https://github.com/telekom/HPOflow/blob/main/hpoflow/optuna_transformers.py):

```bash
$ pip install mlflow GitPython transformers
```

## Support and Feedback

The following channels are available for discussions, feedback, and support requests:

- [open an issue in our GitHub repository](https://github.com/telekom/HPOflow/issues/new/choose)
- [send an e-mail to our open source team](mailto:opensource@telekom.de)

## Contribution

Our commitment to open source means that we are enabling -in fact encouraging- all interested
parties to contribute and become part of our developer community.

Contribution and feedback is encouraged and always welcome. For more information about how to
contribute, as well as additional contribution information, see our
[Contribution Guidelines](https://github.com/telekom/HPOflow/blob/main/CONTRIBUTING.md).
By participating in this project, you agree to abide by its
[Code of Conduct](https://github.com/telekom/HPOflow/blob/main/CODE_OF_CONDUCT.md) at all times.

## Code of Conduct

This project has adopted the [Contributor Covenant](https://www.contributor-covenant.org/)
in version 2.0 as our code of conduct. Please see the details in our
[CODE_OF_CONDUCT.md](https://github.com/telekom/HPOflow/blob/main/CODE_OF_CONDUCT.md).
All contributors must abide by the code of conduct.

## Working Language

We decided to apply _English_ as the primary project language.

Consequently, all content will be made available primarily in English. We also ask all interested
people to use English as language to create issues, in their code (comments, documentation etc.) and
when you send requests to us. The application itself and all end-user facing content will be made
available in other languages as needed.

## Licensing

Copyright (c) 2021 Philip May, Deutsche Telekom AG<br/>
Copyright (c) 2021 Philip May<br/>
Copyright (c) 2021 Timothy Wolff-Piggott

Licensed under the **MIT License** (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License by reviewing the file
[LICENSE](https://github.com/telekom/HPOflow/blob/main/LICENSE) in the repository.
