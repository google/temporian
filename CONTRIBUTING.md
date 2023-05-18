# How to Contribute

This guide describes how to contribute to Temporian, and will help you set up your environment and create your first submission.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License Agreement.

You (or your employer) retain the copyright to your contribution, this simply gives us permission to use and redistribute your contributions as part of the project. Head over to <https://cla.developers.google.com/> to see your current agreements on file or sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one (even if it was for a different project), you probably won't need to do it again.

## Code reviews

All submissions, including submissions by project members, require review. We use GitHub Pull Requests for this purpose. Consult [GitHub Help](https://help.github.com/articles/about-pull-requests/) for more information on using pull requests.

All new contributions must pass all the tests and checks performed by GitHub actions, and any changes to docstrings must respect the [docstring guidelines](docs/docstring_guidelines.md).

## Development

### Environment Setup

Install [Poetry](https://python-poetry.org/), which we use to manage Python dependencies and virtual environments.

Temporian requires Python `3.9.0` or greater. We recommend using [PyEnv](https://github.com/pyenv/pyenv#installation) to install and manage multiple Python versions. Once PyEnv is available, install a supported Python version (e.g. 3.9.6) by running:

```shell
pyenv install 3.9.6
```

After both Poetry and an adequate Python version have been installed, you can proceed to install the virtual environment and the required dependencies. Navigate to the project's root and run:

```shell
pyenv which python | xargs poetry env use
poetry install
```

You can also install the environment in the project's root directory by executing `poetry config virtualenvs.in-project true` before it.

Finally, activate the virtual environment by executing:

```shell
poetry shell
```

### Testing

Install bazel and buildifier (in Mac we recommend installing bazelisk with brew):

```shell
brew install bazelisk
```

Run all tests with bazel:

```shell
bazel test //...:all
```

You can use the Bazel test flag `--test_output=streamed` to see the test logs in realtime.

### Benchmarking and profiling

Benchmarking and profiling of pre-configured scripts is available as follow:

#### Time and memory profiling

```shell
bazel run -c opt benchmark:profile_time -- [name]
bazel run -c opt benchmark:profile_memory -- [name] [-p]
```

where `[name]` is the name of one of the python scripts in
[benchmark/scripts](benchmark/scripts), e.g. `bazel run -c opt benchmark:profile_time -- basic`.

`-p` flag displays memory over time plot instead of line-by-line memory
consumption.

#### Time benchmarking

```shell
bazel run -c opt benchmark:benchmark_time
```

### Running docs server

Live preview your local changes to the documentation with

```shell
mkdocs serve -f docs/mkdocs.yml
```
