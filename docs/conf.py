# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "virtualizarr"
copyright = "2024, Thomas Nicholas"
author = "Thomas Nicholas"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "myst_nb",
    "autodoc2",
    "sphinx.ext.extlinks",
    # "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_togglebutton",
    "sphinx_design",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autodoc2_packages = {
    "path": "../virtualizarr",
    "auto_mode": False,
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "repository_url": "https://github.com/TomNicholas/VirtualiZarr",
    "repository_branch": "main",
    "path_to_docs": "docs",
}

html_logo = "_static/_future_logo.png"

html_static_path = ["_static"]


# issues
# pangeo logo
# dark mode/lm switch
# needs to add api ref
