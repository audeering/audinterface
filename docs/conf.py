from datetime import date

import toml

import audeer


config = toml.load(audeer.path("..", "pyproject.toml"))


# Project -----------------------------------------------------------------
project = config["project"]["name"]
copyright = f"2020-{date.today().year} audEERING GmbH"
author = ", ".join(author["name"] for author in config["project"]["authors"])
version = audeer.git_repo_version()
title = "Documentation"


# General -----------------------------------------------------------------
master_doc = "index"
extensions = []
source_suffix = ".rst"
exclude_patterns = [
    "build",
    "tests",
    "Thumbs.db",
    ".DS_Store",
    "api-src",
]
pygments_style = None
extensions = [
    "jupyter_sphinx",
    "sphinx.ext.napoleon",  # support for Google-style docstrings
    "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinx_copybutton",  # for "copy to clipboard" buttons
    "sphinx_apipages",
]
intersphinx_mapping = {
    "audformat": ("https://audeering.github.io/audformat/", None),
    "audmath": ("https://audeering.github.io/audmath/", None),
    "audobject": ("https://audeering.github.io/audobject/", None),
    "audresample": ("https://audeering.github.io/audresample/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "opensmile": ("https://audeering.github.io/opensmile-python/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "python": ("https://docs.python.org/3/", None),
}
# Disable Gitlab as we need to sign in
linkcheck_ignore = [
    "https://gitlab.audeering.com",
]
# Ignore package dependencies during building the docs
autodoc_mock_imports = [
    "tqdm",
]
autodoc_inherit_docstrings = True

# Reference with :ref:`data-header:Database`
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2

# Do not copy prompot output
copybutton_prompt_text = r">>> |\.\.\. |$ "
copybutton_prompt_is_regexp = True

# HTML --------------------------------------------------------------------
html_theme = "sphinx_audeering_theme"
html_theme_options = {
    "display_version": True,
    "logo_only": False,
    "footer_links": False,
}
html_context = {
    "display_github": True,
}
html_title = title
