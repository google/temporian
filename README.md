# Temporal Feature Processor

**Temporal Feature Processor** (TFP) is a library to pre-process temporal
signals before their use as input features with of-the-shelf tabular machine
learning libraries (e.g., TensorFlow Decision Forests).

Note: Temporal Feature Processor is the placeholder name of this library.

Note: This repository is the temporary location of this library.

## Requirements

Dependencies are managed through [Poetry](https://python-poetry.org/). To
install Poetry, execute the following command:

```shell
curl -sSL https://install.python-poetry.org | python3 -
```

You can verify Poetry was correctly installed by executing:

```shell
poetry --version
```

The environment requires Python version 3.9.0 or greater to be installed. We
recommend using [PyEnv](https://github.com/pyenv/pyenv#installation) to install
and manage multiple Python versions. To install PyEnv, head over to the tool's
[documentation in Github](https://github.com/pyenv/pyenv#installation) and follow the
installation instructions for your operating system.

Once PyEnv is installed, you can download any Python version (e.g. 3.9.6) by
running

```shell
pyenv install 3.9.6
```

After both Poetry and an adequate Python version have been installed, you can
proceed to install the virtual environment and the required dependencies.
Navigate to the project's root directory (where the `pyproject.toml` file is
located) and execute:

```shell
poetry install
```

The virtual environment installation directory will depend on your operating
system as follows:

-   Linux: `$XDG_CACHE_HOME/pypoetry/virtualenvs or
    ~/.cache/pypoetry/virtualenvs`
-   Windows: `%LOCALAPPDATA%\pypoetry/virtualenvs`
-   MacOS: `~/Library/Caches/pypoetry/virtualenvs`

You can also install the environment in the project's root directory by
executing `shell poetry config virtualenvs.in-project true` before running
`poetry install`.

Finally, activate the virtual environment by executing:

```shell
poetry shell
```

## Run all tests

Go to the root `temporal_feature_processor` directory, activate the virtual
environment and execute:

```shell
bazel test //...:all
```

## Credits

This project is a collaboration between Google and Tryolabs.
