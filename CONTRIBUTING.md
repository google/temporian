# Contributing

Temporian is an open source project - we warmly welcome and appreciate any kind of contribution! 🤝🏼

This guide describes how to contribute, and will help you set up your environment and create your first submission.

Check out the open [GitHub issues](https://github.com/google/temporian/issues) to see what we're working on, and what we need help with.
Look especially for the `good first issue` label, which indicates issues that are suitable for new contributors.

If you'd like help or additional guidance to contribute, please join our [Discord](https://discord.gg/nT54yATCTy).

## Code reviews

All submissions, including submissions by project members, require review. We use GitHub Pull Requests for this purpose. Consult [GitHub Help](https://help.github.com/articles/about-pull-requests/) for more information on using pull requests.

All new contributions must pass all the tests and checks performed by GitHub actions, and any changes to docstrings must respect the [docstring guidelines](tools/docstring_guidelines.md).

## Development

### Environment Setup

You can't push directly to our repository so you'll have to create a Fork, visit this [guide](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project) for more information.

After forking the repo you should clone your repository:

```shell
git clone https://github.com/YOUR-USERNAME/temporian
```

And manually install the git hooks:

```shell
cp .git-hooks/* .git/hooks
```

Install [Poetry](https://python-poetry.org/), which we use to manage Python dependencies and virtual environments.

Temporian requires Python `3.9.0` or greater. We recommend using [PyEnv](https://github.com/pyenv/pyenv#installation) to install and manage multiple Python versions. Once PyEnv is available, install a supported Python version (e.g. 3.9.6) by running:

```shell
pyenv install 3.9.6
```

After both Poetry and an adequate Python version have been installed, you can proceed to install the virtual environment and the required dependencies.

Configure poetry to create the virtual environment in the project's root directory (some vscode settings depend on this) by executing:

```shell
poetry config virtualenvs.in-project true
```

Before installing the package you need to install [bazel](https://bazel.build/install) (in Mac we recommend installing bazelisk with brew):

```shell
brew install bazelisk
```

Navigate to the project's root and create a Poetry environment with the correct PyEnv version and all needed dependencies with:

```shell
pyenv which python | xargs poetry env use
poetry install
```

Finally, activate the virtual environment by executing:

```shell
poetry shell
```

### Testing

All tests must pass for your contribution to be accepted.

Run all tests with bazel:

```shell
bazel test --config=linux //...:all --test_output=errors
```

You can use the Bazel test flag `--test_output=streamed` to see the test logs in realtime.

If developing and testing C++ code, the `--compilation_mode=dbg` flag enables additional assertions that are otherwise disabled.

Note that these tests also include docstring examples, using the builtin `doctest` module.
See the [Adding code examples](#adding-code-examples) section for more information.

### Running the documentation server

Live preview your local changes to the documentation with

```shell
mkdocs serve -f docs/mkdocs.yml
```

### Benchmarking and profiling

Benchmarking and profiling of pre-configured scripts is available as follow:

#### Time and memory profiling

```shell
bazel run -c opt --config=linux //benchmark:profile_time -- [name]
bazel run -c opt --config=linux //benchmark:profile_memory -- [name] [-p]
```

where `[name]` is the name of one of the python scripts in
[benchmark/scripts](benchmark/scripts), e.g. `bazel run -c opt --config=linux benchmark:profile_time -- basic`.

`-p` flag displays memory over time plot instead of line-by-line memory
consumption.

#### Time benchmarking

```shell
bazel run -c opt --config=linux //benchmark:benchmark_time
```

### Developing a new operator

We provide a utility script that generates placeholder files, modifies existing ones, and prints needed modifications to develop and make available a new operator. From the project's root, run:

```shell
tools/create_operator.py --operator <name>
```

so for example, to create the `EventSet.map()` operator, you'd run `tools/create_operator.py --operator map`.

Here are some key files you'll need to modify (and write the operator's logic in):

- [temporian/core/event_set_ops.py](temporian/core/event_set_ops.py) or [temporian/**init**.py](temporian/__init__.py), depending on if the operator is available in the `EventSet` class (like `EventSet.since_last()`) or in the global `tp` module (like `tp.glue()`).
- Write the operator's core logic in `temporian/core/operators/<name>.py`.
  - The core logic is that related to the operator's definition in the graph, checks, and normalization during initialization. It doesn't interact with the actual data contained within the `EventSet`.
  - Example: [temporian/core/operators/since_last.py](temporian/core/operators/since_last.py).
- Write the operator's implementation in `temporian/implementation/numpy/operators/<name>.py`.
  - The implementation is what actually executes the operator's logic on an `EventSet`'s data.
  - Example of a Python-only operator: [temporian/implementation/numpy/operators/since_last.py](temporian/implementation/numpy/operators/since_last.py).
  - Example of a C++ operator: [temporian/implementation/numpy/operators/resample.py](temporian/implementation/numpy/operators/resample.py) and [temporian/implementation/numpy_cc/operators/resample.cc](temporian/implementation/numpy_cc/operators/resample.cc).
- Write unit tests for the operator in `temporian/core/operators/test/test_<name>.py`.
  - Example: [temporian/core/operators/test/test_since_last.py](temporian/core/operators/test/test_since_last.py).
- Add the operator to the docs in `docs/src/reference/temporian/operators/<name>.md`.
  - The docs are generated automatically by [mkdocstrings](https://mkdocstrings.github.io/python/) from the operator's docstring.
  - Example: [docs/src/reference/temporian/operators/since_last.md](docs/src/reference/temporian/operators/since_last.md).

Groups of operator with a similar implementation as grouped together. For instance, `temporian/core/operators/window` contains moving window operators (e.g., `EventSet.simple_moving_average()`) and `temporian/core/operators/binary` contains operators taking two features as input (e.g. `EventSet.subtract()`).

Read the script's output to see in detail all other files that need to be modified to finish setting up the operator!

### Adding code examples

Any doctest code examples in `temporian/*.py` or `docs/*.md`, will be executed and tested using the python's built-in [doctest](https://docs.python.org/3/library/doctest.html) module.

For example, the following piece of code would be executed, and the outputs must match the expected result indicated:

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
- The `...` inside the expected result is used to match anything. Here, the exact timestamps and the latest line (which includes memory usage information) don't need exact match.

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
ValueError: ... corresponding features should have the same dtype. ...

```

Finally, note that globals like `tp`, `pd` and `np` are always included in the execution context, no need to import them.

To check if your examples are correct, you may run:

```shell
# Test anything in temporian/*.py and docs/*.md
bazel test --config=linux //temporian/test:doc_test --test_output=streamed
```

In case of unexpected outputs, the result is printed and compared to the expected values, so that they can be fixed.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License Agreement.

You (or your employer) retain the copyright to your contribution, this simply gives us permission to use and redistribute your contributions as part of the project. Head over to <https://cla.developers.google.com/> to see your current agreements on file or sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one (even if it was for a different project), you probably won't need to do it again.
