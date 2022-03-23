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
import sys, os
from importlib import import_module

# import os
# import sys
sys.path.insert(0, os.path.abspath("../../"))
# sys.path.append(os.path.abspath("."))
# -- Project information -----------------------------------------------------

project = "cosmoglobe"
copyright = "2021, Metin San"
author = "Metin San"

# The full version, including alpha/beta/rc tags
release = "1.0.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autosectionlabel",
    # "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "sphinx_autodoc_typehints",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

napoleon_use_param = True
napoleon_numpy_docstring=True
napoleon_include_init_with_doc=True
napoleon_attr_annotations=True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

# Using pydata theme with custom numpy css
html_theme = "pydata_sphinx_theme"

html_logo = "_static/cosmoglobe.svg"

html_favicon = "_static/favicon/favicon-96x96.png"

html_theme_options = {
    "logo_link": "index",
    "collapse_navigation": True,
    "github_url": "https://github.com/Cosmoglobe/Cosmoglobe",
}

html_static_path = ["_static/"]

html_css_files = ["css/cosmoglobe.css"]

source_suffix = [".rst"]

# import_module('cosmoglobe')
# package = sys.modules['cosmoglobe']
# version = package.__version__.split("-", 1)[0]
# release = package.__version__

# today_fmt = "%B %d, %Y"
# html_last_updated_fmt = "%b %d, %Y"

# import glob

# autosummary_generate = glob.glob("*.rst")
