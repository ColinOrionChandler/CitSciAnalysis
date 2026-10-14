"""Sphinx configuration for CitSciAnalysis."""
project = "CitSciAnalysis"
author = "Will Burris, Colin Chandler, and Chad Trujillo"
copyright = "2026, CitSciAnalysis contributors"
extensions = ["autoapi.extension", "myst_parser", "sphinx.ext.napoleon"]
autoapi_dirs = ["../src/citsci_analysis"]
autoapi_ignore = ["*/_version.py"]
autoapi_options = ["members", "undoc-members", "show-inheritance", "show-module-summary"]
autoapi_keep_files = False
autoapi_add_toctree_entry = False
napoleon_numpy_docstring = True
html_theme = "sphinx_rtd_theme"
exclude_patterns = ["_build", "autoapi"]
