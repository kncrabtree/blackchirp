# Configuration file for the Sphinx documentation builder.
#
# Full reference:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re
import subprocess
import sys
from datetime import date

# Make the python/blackchirp package importable for autodoc.
sys.path.insert(0, os.path.abspath('../../python/blackchirp/src'))

# Run Doxygen so breathe has up-to-date XML. The Doxyfile in this directory
# is the ReadTheDocs variant maintained by cmake/BlackchirpDocumentation.cmake;
# CMake builds run Doxygen via the `doxygen` target as well, so this call is
# idempotent in either environment.
subprocess.call('doxygen Doxyfile', shell=True)


# -- Helpers ----------------------------------------------------------------

def _read_version_from_cmake():
    """Parse BC_*_VERSION values from the top-level CMakeLists.txt.

    The CMake project() declaration carries the canonical
    major/minor/patch numbers; the release suffix (e.g. "alpha") is
    declared separately. Returning ("", "") is safe — Sphinx falls back
    to the strings configured below.
    """
    cmake_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'CMakeLists.txt')
    )
    try:
        with open(cmake_path, 'r', encoding='utf-8') as fh:
            text = fh.read()
    except OSError:
        return ('', '')

    project_match = re.search(
        r'project\s*\(\s*Blackchirp\b[^)]*?VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)',
        text,
        re.DOTALL,
    )
    base = project_match.group(1) if project_match else ''

    release_match = re.search(
        r'set\s*\(\s*BC_RELEASE_VERSION\s+"([^"]+)"',
        text,
    )
    release = release_match.group(1) if release_match else ''

    full = base if not release else f'{base}-{release}'
    return (base, full)


# -- Project information ----------------------------------------------------

project = 'Blackchirp'
author = 'Kyle Crabtree'
copyright = f'{date.today().year}, {author}'

_short_version, _full_version = _read_version_from_cmake()
# Sphinx convention: `version` is the short X.Y string used in the sidebar
# title; `release` is the full version including any pre-release suffix.
version = _short_version or '2.0.0'
release = _full_version or '2.0.0-alpha'


# -- General configuration --------------------------------------------------

extensions = [
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autodoc',
    'sphinx.ext.extlinks',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'nbsphinx_link',
    'breathe',
    'sphinxcontrib.mermaid',
]

# `:commit:` role rendered as a short hash linked to the GitHub commit page.
# Use the short hash as the role argument; GitHub resolves any prefix length
# to the canonical commit URL.
extlinks = {
    'commit': (
        'https://github.com/kncrabtree/blackchirp/commit/%s',
        '%s',
    ),
}

# sphinx_rtd_theme is always light, but sphinxcontrib-mermaid auto-detects a
# dark Mermaid theme from the OS prefers-color-scheme media query. Force the
# light variant by tagging <html> with the 'light' class before the Mermaid
# ESM module evaluates; the package's heuristic short-circuits on that class.
html_js_files = ['mermaid_force_light.js']

autosectionlabel_prefix_document = True
autodoc_mock_imports = ['pandas', 'scipy', 'numpy']
nbsphinx_execute = 'never'

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output ------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_favicon = '_static/favicon.png'


# -- Options for breathe ----------------------------------------------------

# Doxygen XML lands in `xml/` next to this conf.py for both ReadTheDocs and
# CMake builds — the CMake target also writes an absolute-path copy under
# the build tree, but breathe only needs the source-relative path.
breathe_projects = {'Blackchirp': 'xml'}
breathe_default_project = 'Blackchirp'
breathe_default_members = (
    'members',
    'protected-members',
    'private-members',
    'undoc-members',
)
breathe_show_include = False
