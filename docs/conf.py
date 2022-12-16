from datetime import date
import os
import shutil
import subprocess

import audeer


# Project -----------------------------------------------------------------
project = 'audinterface'
copyright = f'2020-{date.today().year} audEERING GmbH'
author = 'Johannes Wagner, Hagen Wierstorf, Andreas Triantafyllopoulos'
# The x.y.z version read from tags
try:
    version = subprocess.check_output(['git', 'describe', '--tags',
                                       '--always'])
    version = version.decode().strip()
except Exception:
    version = '<unknown>'
title = f'{project} Documentation'


# General -----------------------------------------------------------------
master_doc = 'index'
extensions = []
source_suffix = '.rst'
exclude_patterns = [
    'build',
    'tests',
    'Thumbs.db',
    '.DS_Store',
    'api-src',
]
templates_path = ['_templates']
pygments_style = None
extensions = [
    'jupyter_sphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # support for Google-style docstrings
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'sphinx_copybutton',  # for "copy to clipboard" buttons
]
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'audformat': ('https://audeering.github.io/audformat/', None),
    'audobject': ('https://audeering.github.io/audobject/', None),
    'audresample': ('https://audeering.github.io/audresample/', None),
    'opensmile': ('https://audeering.github.io/opensmile-python/', None),
}
# Disable Gitlab as we need to sign in
linkcheck_ignore = [
    'https://gitlab.audeering.com',
]
# Ignore package dependencies during building the docs
autodoc_mock_imports = [
    'tqdm',
]
autodoc_inherit_docstrings = True

# Reference with :ref:`data-header:Database`
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2

# Do not copy prompot output
copybutton_prompt_text = r'>>> |\.\.\. |$ '
copybutton_prompt_is_regexp = True

# Disable auto-generation of TOC entries in the API
# https://github.com/sphinx-doc/sphinx/issues/6316
toc_object_entries = False

# HTML --------------------------------------------------------------------
html_theme = 'sphinx_audeering_theme'
html_theme_options = {
    'display_version': True,
    'logo_only': False,
    'footer_links': False,
}
html_context = {
    'display_github': True,
}
html_title = title

# Copy API (sub-)module RST files to docs/api/ folder ---------------------
audeer.mkdir('api')
api_src_files = audeer.list_file_names('api-src')
api_dst_files = [
    audeer.path('api', os.path.basename(src_file))
    for src_file in api_src_files
]
for src_file, dst_file in zip(api_src_files, api_dst_files):
    shutil.copyfile(src_file, dst_file)
