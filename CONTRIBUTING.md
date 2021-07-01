# Contributing

## Code of conduct

All members of the project community must abide by the [Contributor Covenant v2.0](CODE_OF_CONDUCT.md).
Only by respecting each other can we develop a productive, collaborative community.
Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting 
[opensource@telekom.de](mailto:opensource@telekom.de) and/or a project maintainer.

We appreciate your courtesy of avoiding political questions here. Issues which are not related to 
the project itself will be closed by our community managers.

## Engaging in our project

We use GitHub to manage reviews of pull requests.
* If you are a new contributor, see: [Steps to Contribute](#steps-to-contribute)
* If you have a trivial fix or improvement, go ahead and create a pull request, 
  addressing (with `@...`) a suitable maintainer of this repository (see [Code Owners](#code-owners)).
* If you plan to do something more involved, please reach out to us and send an [email](mailto:opensource@telekom.de). 
  This will avoid unnecessary work and surely give you and us a good deal of inspiration.
* Relevant coding [style guidelines](#style-guidelines) are available in this document.

## Steps to Contribute

Should you wish to work on an issue, please claim it first by commenting
on the GitHub issue that you want to work on. This is to prevent duplicated
efforts from other contributors on the same issue.

If you have questions about one of the issues, please comment on them, 
and one of the maintainers will clarify.

We kindly ask you to follow the [Pull Request Checklist](#Pull-Request-Checklist) 
to ensure reviews can happen accordingly.

## Contributing Code

You are welcome to contribute code in order to fix a bug or to implement a new feature.
The following rules governs code contributions:
* Contributions must be licensed under the [MIT license](LICENSE)
* Newly created files must be opened by the following file header and a
  blank line.

```python
# Copyright (c) <year> <your_name>[, <your_organization>]
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

```

## Contributing Documentation

You are welcome to contribute documentation to the project.
The following rule governs documentation contributions:
Contributions must be licensed under the same license as code, the [MIT license](LICENSE).

## Pull Request Checklist

* Branch from the master branch and, if needed, rebase to the current master branch 
  before submitting your pull request. If it doesn't merge cleanly with master you 
  may be asked to rebase your changes.
* Commits should be as small as possible while ensuring that each commit is correct 
  independently (i.e., each commit should compile and pass tests).
* Test your changes as thoroughly as possible before you commit them. Preferably, 
  automate your test by unit/integration tests. If tested manually, provide information 
  about the test scope in the PR description (e.g. “Test passed: Upgrade version from 
  0.42 to 0.42.23.”).
* To differentiate your PR from PRs ready to be merged and to avoid duplicated work,
  please prefix the title with [WIP].
* If your pull request is not getting reviewed or you need a specific person to review it, 
  you can @-reply a reviewer asking for a review in the pull request or a comment, or you 
  can ask for a review by contacting us via [email](mailto:opensource@telekom.de).
* Post review:
  * If a review requires you to change your commit(s), please test the changes again.
  * Amend the affected commit(s) and force push onto your branch.
  * Set respective comments in your GitHub review to resolved.
  * Create a general PR comment to notify the reviewers that your amendments are ready for 
    another round of review.

## Issues and Planning

* We use GitHub issues to track bugs and enhancement requests.
* Please provide as much context as possible when you open an issue. 
  The information you provide must be comprehensive enough to reproduce 
  that issue for the assignee. Therefore, contributors may use but aren't 
  restricted to the issue template provided by the project maintainers.
* When creating an issue, try using one of our issue templates which 
  already contain some guidelines on which content is expected to process 
  the issue most efficiently. If no template applies, you can of course 
  also create an issue from scratch.

## Style Guidelines

- The code must be compatible with Python 3.6 and higher.
- max line length is 99
- Use the [Google docstring format](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings).
  This is integrated with [Sphinx](https://www.sphinx-doc.org/) using the 
  [napoleon extension](https://sphinxcontrib-napoleon.readthedocs.io/). 

## Release Checklist

- Do all tests pass?
- Did we change or add dependencies?
  - update `install_requires` in `setup.py`?
  - update `extras_require` in `setup.py`?
  - update `THIRD-PARTY-NOTICES`?
- Do we need to add persons or organizations to the `LICENSE` file?
- Is the documentation up to date?
- Did we change the Python version requirements?
  - update `python_requires` in `setup.py`
  - update `target-version` in `pyproject.toml`
  - update `classifiers - Programming Language :: Python` in `setup.py`
- other checks
  - does `classifiers` (Development Status) need an update?
- if we want to do a full release change version number in `version.py`
- if we want to do a development release no version change is needed
- create a new release in GitHub  
- bump version number in `version.py` to a new `.devx` version

## Code Owners
[@PhilipMay](https://github.com/PhilipMay) - general documentation, GitHub actions, 
  `optuna_mlflow.py`, `optuna.py`, everything else
