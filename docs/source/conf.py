# Configuration file for the Sphinx documentation builder.

"""

Steps to generate documentation:
1) cd to /docs
2) run sphinx-apidoc -f -o /source ../masterplan_tools/   - generates project structure
3) run make html (or .\make html if you don't have make available globally)    - builds html with sphinx structure
4) newly built documentation will be in /docs/build/html

"""


import os
import sys

sys.path.insert(0, os.path.abspath("../../"))
sys.path.append(os.path.abspath(os.path.join(__file__, "../../masterplan_tools")))


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
    "numpydoc",
]

templates_path = ["_templates"]
exclude_patterns = []


html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
