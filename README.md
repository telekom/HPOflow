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

## Table of Contents

- [Maintainers](#maintainers)
- [Installation](#installation)
- [Support and Feedback](#support-and-feedback)
- [Reporting Security Vulnerabilities](#reporting-security-vulnerabilities)
- [Contribution](#contribution)
- [Code of Conduct](#code-of-conduct)
- [Licensing](#licensing)

## Maintainers

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
It can be installed with pip:

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

## Reporting Security Vulnerabilities

This project is built with security and data privacy in mind to ensure your data is safe.
We are grateful for security researchers and users reporting a vulnerability to us, first.
To ensure that your request is handled in a timely manner and non-disclosure of vulnerabilities
can be assured, please follow the below guideline.

**Please do not report security vulnerabilities directly on GitHub.
GitHub Issues can be publicly seen and therefore would result in a direct disclosure.**

Please address questions about data privacy, security concepts,
and other media requests to the [opensource@telekom.de](mailto:opensource@telekom.de) mailbox.

## Contribution

Our commitment to open source means that we are enabling - in fact encouraging - all interested
parties to contribute and become part of our developer community.

Contribution and feedback is encouraged and always welcome. For more information about how to
contribute, as well as additional contribution information, see our
[Contribution Guidelines](https://github.com/telekom/HPOflow/blob/main/CONTRIBUTING.md).

## Code of Conduct

This project has adopted the [Contributor Covenant](https://www.contributor-covenant.org/)
as our code of conduct. Please see the details in our
[Contributor Covenant Code of Conduct](https://github.com/telekom/HPOflow/blob/main/CODE_OF_CONDUCT.md).
All contributors must abide by the code of conduct.

## Licensing

Copyright (c) 2021 Philip May, Deutsche Telekom AG<br/>
Copyright (c) 2021 Philip May<br/>
Copyright (c) 2021 Timothy Wolff-Piggott

Licensed under the **MIT License** (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License by reviewing the file
[LICENSE](https://github.com/telekom/HPOflow/blob/main/LICENSE) in the repository.
