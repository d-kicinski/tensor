# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
from pathlib import Path
import os
import sys
import subprocess
import sphinx_rtd_theme

DIR = Path(__file__).parent.resolve()

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'tensor'
copyright = '2021, Dawid Kiciński'
author = 'Dawid Kiciński'

# The full version, including alpha/beta/rc tags
release = '0.2.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "breathe",
    "sphinx_rtd_theme",
    ]

source_suffix = {
    '.rst': 'restructuredtext',
}

breathe_projects = {"tensor": ".build/doxygenxml/"}
breathe_default_project = "tensor"
breathe_domain_by_extension = {"hpp": "cpp"}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


def generate_doxygen_xml(app):
    build_dir = os.path.join(app.confdir, ".build")
    if not os.path.exists(build_dir):
        os.mkdir(build_dir)

    try:
        subprocess.call(["doxygen", "--version"])
        retcode = subprocess.call(["doxygen"], cwd=app.confdir)
        if retcode < 0:
            sys.stderr.write("doxygen error code: {}\n".format(-retcode))
    except OSError as e:
        sys.stderr.write("doxygen execution failed: {}\n".format(e))


def prepare(app):
    with open(DIR.parent / "README.rst") as f:
        contents = f.read()

    with open(DIR / "readme.rst", "w") as f:
        f.write(contents)


def clean_up(app, exception):
    (DIR / "readme.rst").unlink()


def setup(app):

    # Add hook for building doxygen xml when needed
    app.connect("builder-inited", generate_doxygen_xml)

    # Copy the readme in
    app.connect("builder-inited", prepare)

    # Clean up the generated readme
    app.connect("build-finished", clean_up)
