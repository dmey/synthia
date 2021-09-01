import os
import sys
from pathlib import Path
import shutil
import inspect

PROJ_DIR = Path.cwd().parent
sys.path.insert(0, str(PROJ_DIR.resolve()))


def copy_content(f_read: Path, f_write: Path, rebuild=False) -> None:
    if rebuild:
        with open(f_read) as fr, open(f_write, 'w') as fw:
            if f_write.stem == 'license':
                header = '# Copyright & License\n\n'
                fw.write(header)
            content = fr.readlines()
            fw.writelines(content)

copy_content(PROJ_DIR / 'CONTRIBUTING.md', PROJ_DIR / 'docs' / 'contributing.md', True)
copy_content(PROJ_DIR / 'DEVELOP.md', PROJ_DIR / 'docs' / 'develop.md', True)
copy_content(PROJ_DIR / 'LICENSE.txt', PROJ_DIR / 'docs' / 'license.md', True)


def copy_overview(f_read: Path, f_write: Path, rebuild=False) -> None:
    if rebuild:
        with open(f_read) as fr, open(f_write, 'w') as fw:
            parse = False
            content = fr.readlines()
            fw.write('# Overview\n')
            for line in content: 
                if line.startswith('Synthia is'): 
                    parse = True 
                if line.startswith('##'): 
                    parse = False 
                if parse:
                    fw.write(line)
    # Copy the figures as well as relative paths are left unchnaged
    ASSETS_SOURCE_DIR = PROJ_DIR / 'assets'
    ASSETS_DEST_DIR = PROJ_DIR / 'docs' / 'assets'
    if ASSETS_DEST_DIR.exists():
        shutil.rmtree(ASSETS_DEST_DIR)
    shutil.copytree(ASSETS_SOURCE_DIR, ASSETS_DEST_DIR)

copy_overview(PROJ_DIR / 'README.md', PROJ_DIR / 'docs' / 'overview.md', True)

project = 'synthia'
copyright = '2020 D. Meyer and T. Nagler'
author = 'D. Meyer and T. Nagler'
release = '1.1.0'

html_context = {
  'display_github': True,
  'github_user': 'dmey',
  'github_repo': 'synthia',
  'github_version': 'master/docs/',
}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
    'sphinxcontrib.bibtex',
    'sphinx_copybutton',
    "myst_parser"
]

# Autodoc settings
autodoc_typehints = 'none' # 'signature'

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = False

copybutton_prompt_text = ">>> "


nbsphinx_execute = 'always'
# Some notebooks take longer than the default 30 s limit.
nbsphinx_timeout = 120

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
if os.environ.get('SKIP_NB') == '1':
    exclude_patterns.append('examples/*')

html_logo = "../assets/img/logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'collapse_navigation': False,
    'display_version': False,
    "logo_only": True
}

def skip_constructors_without_args(app, what, name, obj, would_skip, options):
    if name == '__init__':
        func = getattr(obj, '__init__')
        spec = inspect.getfullargspec(func)
        return not spec.args and not spec.varargs and not spec.varkw and not spec.kwonlyargs
    return would_skip

def setup(app):
    app.connect("autodoc-skip-member", skip_constructors_without_args)
