# Copyright (c) 2021 Philip May
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""HPOflow main package."""

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter


# Versioning follows the Semantic Versioning Specification https://semver.org/ and
# PEP 440 -- Version Identification and Dependency Specification: https://www.python.org/dev/peps/pep-0440/  # noqa: E501
__version__ = "0.1.4"

_import_structure = {
    "mlflow": [
        "normalize_mlflow_entry_name",
        "normalize_mlflow_entry_names_in_dict",
        "check_repo_is_dirty",
    ],
    "optuna": ["SignificanceRepeatedTrainingPruner"],
    "optuna_mlflow": ["OptunaMLflow"],
    "optuna_transformers": ["OptunaMLflowCallback"],
    "utils": ["func_no_exception_caller"],
}

# Direct imports for type-checking
if TYPE_CHECKING:
    from hpoflow.mlflow import (  # noqa: F401
        check_repo_is_dirty,
        normalize_mlflow_entry_name,
        normalize_mlflow_entry_names_in_dict,
    )
    from hpoflow.optuna import SignificanceRepeatedTrainingPruner  # noqa: F401
    from hpoflow.optuna_mlflow import OptunaMLflow  # noqa: F401
    from hpoflow.optuna_transformers import OptunaMLflowCallback  # noqa: F401
    from hpoflow.utils import func_no_exception_caller  # noqa: F401
else:
    sys.modules[__name__] = LazyImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
        extra_objects={"__version__": __version__},
    )
