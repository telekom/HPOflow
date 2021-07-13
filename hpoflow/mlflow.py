# Copyright (c) 2021 Philip May, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""MLflow only functionality and tools."""

import logging
import os
import re
from typing import Any, Dict

import git
from mlflow.tracking.context.default_context import _get_main_file


_logger = logging.getLogger(__name__)
_normalize_mlflow_entry_name_re = re.compile(r"[^a-zA-Z0-9-._ /]")


def normalize_mlflow_entry_name(name: str) -> str:
    """Normalize a MLflow entry name."""
    name = name.replace("Ä", "Ae")
    name = name.replace("Ö", "Oe")
    name = name.replace("Ü", "Ue")
    name = name.replace("ä", "ae")
    name = name.replace("ö", "oe")
    name = name.replace("ü", "ue")
    name = name.replace("ß", "ss")
    name = re.sub(_normalize_mlflow_entry_name_re, "_", name)
    return name


def normalize_mlflow_entry_names_in_dict(dct: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the keys of a MLflow entry dict."""
    keys = list(dct.keys()).copy()  # must create a copy do keys do not change while iteration
    for key in keys:
        dct[normalize_mlflow_entry_name(key)] = dct.pop(key)
    return dct


def check_repo_is_dirty() -> None:
    """Check if the repository is considered dirty (see :meth:`git.repo.base.Repo.is_dirty`).

    By default it will react like a git-status without untracked files, hence it is dirty if the
    index or the working copy have changes.

    Raises:
        RuntimeError: If the repository is considered dirty.
    """
    path = _get_main_file()
    if os.path.isfile(path):
        path = os.path.dirname(path)
    repo = git.Repo(path, search_parent_directories=True)
    if repo.is_dirty():
        error_message = "Git repository '{}' is dirty!".format(path)
        _logger.error(error_message)
        raise RuntimeError(error_message)
    _logger.info("Git repository '%s' is clean.", path)
