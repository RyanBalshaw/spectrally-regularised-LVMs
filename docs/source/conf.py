"""
Configuration file for the Sphinx documentation builder.
"""
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information

project = "spectrally-regularised-LVMs"
copyright = "2023 - present, Ryan Balshaw"
author = "Ryan Balshaw"

release = "0.1"
version = "0.1.1"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",  # To generate autodocs
    "sphinx.ext.mathjax",  # autodoc with maths
    "sphinx.ext.napoleon",  # For auto-doc configuration
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "numpydoc",
    "autodocsumm",
]

napoleon_numpy_docstring = True  # Turn on numpydoc strings

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]
intersphinx_disabled_reftypes = ["*"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "furo"  # "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"

# autodocumentation properties
autodoc_default_options = {"autosummary": True}

# Auto-section properties
autosectionlabel_prefix_document = True
