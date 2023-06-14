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

Note that these tests also include docstring examples, using the builtin `doctest` module.
See the [Adding code examples](#adding-code-examples) section for more information.

### Benchmarking and profiling

Benchmarking and profiling of pre-configured scripts is available as follow:

#### Time and memory profiling

```shell
bazel run -c opt //benchmark:profile_time -- [name]
bazel run -c opt //benchmark:profile_memory -- [name] [-p]
```

where `[name]` is the name of one of the python scripts in
[benchmark/scripts](benchmark/scripts), e.g. `bazel run -c opt benchmark:profile_time -- basic`.

`-p` flag displays memory over time plot instead of line-by-line memory
consumption.

#### Time benchmarking

```shell
bazel run -c opt //benchmark:benchmark_time
```

### Running docs server

Live preview your local changes to the documentation with

```shell
mkdocs serve -f docs/mkdocs.yml
```

### Adding code examples

Any code examples that are included in the docstrings of api-facing modules,
or in markdown files under the [docs/](docs/) directory,
will be executed and tested using the python's
built-in [doctest](https://docs.python.org/3/library/doctest.html) module.

For example, the following piece of code would be executed, and the outputs
must match the expected result indicated:

```python
>>> evset = tp.event_set(
... 	timestamps=["2020-01-01", "2020-02-02"],
... )
>>> print(evset)
indexes: []
features: []
events:
     (2 events):
        timestamps: [...]
...

```

Note from this example:

- If the `>>>` indicator is not present, the code will not be run or tested.
- Multi-line statements need a preceding `...` instead of `>>>`.
- All the lines immediately following `>>>` or `...` and before a blank line, are the expected outputs.
- You should always leave a blank line before closing the code block, to indicate the end of the test.
- The `...` inside the expected result is used to match anything. Here, the exact timestamps and the latest line (which includes memory usage information).

You cannot use `...` in the first matching line to ignore the whole output (it's ambiguous with multi-lines).
In that case, you may use the `SKIP` flag as follows:

```
>>> print("hello")  # doctest:+SKIP
This result doesn't need to match
```

Exceptions can also be expected, but it's better to avoid being too specific with the expected result:

```python
>>> node["f1"] + node["f2"]
Traceback (most recent call last):
    ...
ValueError: ... corresponding features (with the same index) should have the same dtype. ...

```

Finally, note that globals like `tp`, `pd` and `np` are always included in the execution context, no need to import them.

To check if your examples are correct, you may run:

```shell
# Test all examples in code docstrings (only api-facing modules)
bazel test //temporian/test:docstring_test --test_output=streamed

# Test all examples in /docs/*.md
bazel test //docs/code_examples_test --test_output=streamed
```

In case of unexpected outputs, the result is printed and compared to the expected values, so that they can be fixed.
