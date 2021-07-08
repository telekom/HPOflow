# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "HPOflow"
copyright = "2021, Philip May, Deutsche Telekom AG"
author = "Philip May"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_rtd_theme",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "recommonmark",
    "sphinx.ext.intersphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- additional settings -------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "transformers": ("https://huggingface.co/transformers/", None),
    "optuna": ("https://optuna.readthedocs.io/en/stable/", None),
    "mlflow": ("https://www.mlflow.org/docs/latest/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

html_logo = "imgs/1c-logo.png"

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    "css/custom.css",
]

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# add github link
html_context = {
    "display_github": True,
    "github_user": "telekom",
    "github_repo": "HPOflow",
    "github_version": "main/docs/source/",
}

source_suffix = [".rst", ".md"]

# -- autosummary config -------------------------------------------------
autosummary_generate = True
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": True,
    "special-members": None,
}

# True to convert the type definitions in the docstrings as references.
napoleon_preprocess_types = True
