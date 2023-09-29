# Configuration file for the Sphinx documentation builder.

"""

Steps to generate documentation:
1) cd to /build_docs
2) run sphinx-apidoc -f -o /source ../blocksnet/   - generates project structure
3) run make html (or .\make html if you don't have make available globally)    - builds html with sphinx structure
4) newly built documentation will be in /build_docs/build/html

"""


import os
import pathlib
import sys

sys.path.insert(0, os.path.abspath(str(pathlib.Path(__file__).resolve().parent.parent)))
sys.path.append(os.path.abspath(str(pathlib.Path(__file__).resolve().parent.parent / "blocksnet")))


# TODO: change properties to correct values
project = "Masterplanning"
copyright = "2023, open-source"
author = "open-source"


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "numpydoc",
]

templates_path = ["_templates"]
exclude_patterns = []


html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
