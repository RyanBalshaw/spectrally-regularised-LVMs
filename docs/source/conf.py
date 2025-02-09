"""
MIT License
-----------

Copyright (c) 2023 Ryan Balshaw

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

--------------------------------------------------------

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
