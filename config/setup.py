import setuptools
from setuptools.dist import Distribution

_VERSION = "0.7.0"


class _BinaryDistribution(Distribution):
    """This class is needed in order to create OS specific wheels."""

    def is_pure(self):
        return False

    def has_ext_modules(self):
        return True


install_requires = [
    "absl-py>=1.3.0,<2.0.0",
    "matplotlib>=3.7.1,<4.0.0",
    "pandas>=1.5.2",
    "protobuf>=3.20.3",
]

extras_require = {"beam": ["apache-beam>=2.48.0,<3.0.0"]}

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup_kwargs = {
    "name": "temporian",
    "version": _VERSION,
    "description": (
        "Temporian is a Python package for feature engineering of temporal"
        " data, focusing on preventing common modeling errors and providing a"
        " simple and powerful API, a first-class iterative development"
        " experience, and efficient and well-tested implementations of common"
        " and not-so-common temporal data preprocessing functions."
    ),
    "long_description": long_description,
    "long_description_content_type":"text/markdown",
    "author": (
        "Mathieu Guillame-Bert, Braulio Ríos, Guillermo Etchebarne, Ian"
        " Spektor, Richard Stotz"
    ),
    "author_email": "gbm@google.com",
    "maintainer": "Mathieu Guillame-Bert",
    "maintainer_email": "gbm@google.com",
    "url": "https://github.com/google/temporian",
     "project_urls":{
        "Documentation": "https://https://temporian.readthedocs.io",
        "Source": "https://github.com/google/temporian.git",
        "Tracker": (
            "https://github.com/google/temporian/issues"
        ),
    },
    "packages": setuptools.find_packages(),
    "install_requires": install_requires,
    "extras_require": extras_require,
    "python_requires": ">=3.9,<3.12",
    "license": "Apache 2.0",
    "include_package_data": True,
    "distclass": _BinaryDistribution,
}

setuptools.setup(**setup_kwargs)
