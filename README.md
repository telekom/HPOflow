# HPOflow

The goal of this project is to provide tools for [Optuna](https://optuna.readthedocs.io/), 
[MLflow](https://www.mlflow.org/docs/latest/index.html) and the integration of both.

The main components are:
- `hpoflow.optuna_mlflow.OptunaMLflow`: A wrapper to log to Optuna and MLflow at the same time.
- `hpoflow.optuna.SignificanceRepeatedTrainingPruner`: An 
  [Optuna Pruner](https://optuna.readthedocs.io/en/stable/reference/pruners.html) 
  to use statistical significance to prune repeated trainings like in a cross validation.

## Development

- The code must be compatible with Python 3.6 or higher
- From the user's point of view we recommend Python 3.8 or higher - then you are compatible with 
  TorchElastic (Torch Distributed Elastic) and more long-lived

_TBD_

## Code of Conduct

This project has adopted the [Contributor Covenant](https://www.contributor-covenant.org/) 
in version 2.0 as our code of conduct. Please see the details in our 
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). All contributors must abide by the code of conduct.

## Working Language

We decided to apply _English_ as the primary project language.  

Consequently, all content will be made available primarily in English. We also ask all interested 
people to use English as language to create issues, in their code (comments, documentation etc.) and 
when you send requests to us. The application itself and all end-user facing content will be made 
available in other languages as needed.

## Documentation

_TBD_

## Support and Feedback

The following channels are available for discussions, feedback, and support requests:

| Type                     | Channel                                                |
| ------------------------ | ------------------------------------------------------ |
| **Issues**   | <a href="/../../issues/new/choose" title="General Discussion"><img src="https://img.shields.io/github/issues/telekom/repobasics?style=flat-square"></a> </a>   |
| **Other Requests**    | <a href="mailto:opensource@telekom.de" title="Email Open Source Team"><img src="https://img.shields.io/badge/email-Open%20Source%20Team-green?logo=mail.ru&style=flat-square&logoColor=white"></a>   |

## How to Contribute

Contribution and feedback is encouraged and always welcome. For more information about how to 
contribute, the project structure, as well as additional contribution information, see our 
[Contribution Guidelines](./CONTRIBUTING.md). By participating in this project, you agree to 
abide by its [Code of Conduct](./CODE_OF_CONDUCT.md) at all times.

## Contributors

Our commitment to open source means that we are enabling -in fact encouraging- all interested 
parties to contribute and become part of its developer community.

## Licensing

For licensing information, see the [MIT license file](LICENSE).
