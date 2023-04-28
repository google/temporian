![Temporian logo](docs/assets/banner.png)

![tests](https://github.com/google/temporian/actions/workflows/test.yaml/badge.svg) ![formatting](https://github.com/google/temporian/actions/workflows/formatting.yaml/badge.svg)

**Temporian** is a library to pre-process temporal signals before their use as input features with off-the-shelf tabular machine learning libraries (e.g., TensorFlow Decision Forests, scikit-learn).

## Usage Example

A minimal end-to-end run looks as follows:

```python
import temporian as tp

# Load the data
event_data = tp.read_event("path/to/data.csv")
event = event_data.node()

# Create Simple Moving Average feature
sma = tp.simple_moving_average(
    input=event,
    window_length=tp.day(5),
)

# Create Lag feature
lag = tp.lag(
    input=event,
    lag=tp.week(1),
)

# Glue features
output_event = tp.glue(event, sma)
output_event = tp.glue(output_event, lag)


# Execute pipeline and get results
output_event = tp.evaluate(
    output_event,
    input_data={
        event: event_data,
    },
)

```

> **Warning**: The library is still under construction. This example usage is what we are aiming to build in the short term.

## Supported Features

Temporian currently supports the following features for pre-processing your temporal data:

- **Simple Moving Average:** calculates the average value of each feature over a specified time window.
- **Lag:** creates new features by shifting the time series data backwards in time by a specified period.
- **Arithmetic Operations:** allows you to perform arithmetic operations (such as addition, subtraction, multiplication, and division) on time series data, between different events.
- More features coming soon!

## Environment Setup

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
running:

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

You can also install the environment in the project's root directory by
executing `poetry config virtualenvs.in-project true` before it.

Finally, activate the virtual environment by executing:

```shell
poetry shell
```

## Testing

Install bazel and buildifier (in Mac we recommend installing bazelisk with brew):

```shell
brew install bazelisk
```

Run all tests with bazel:

```shell
bazel test //...:all
```

> **Note**: You can use the Bazel test flag `--test_output=streamed` to see the test logs in realtime.

## Benchmarking and profiling

Benchmarking and profiling of pre-configured scripts is available as follow:

**Time and memory profiling**

```shell
bazel run -c opt benchmark:profile_time -- [name]
bazel run -c opt benchmark:profile_memory -- [name] [-p]
```

where `[name]` is the name of one of the python scripts in
[benchmark/scripts](benchmark/scripts), e.g. `bazel run -c opt benchmark:profile_time -- basic`.

`-p` flag displays memory over time plot instead of line-by-line memory
consumption.

**Time benchmarking**

```shell
bazel run -c opt benchmark:benchmark_time
```

Example of results:

```
================================================================
Name                              Wall time (s)    CPU time (s)
================================================================
from_dataframe:100                   0.01601       0.01600
from_dataframe:10000                 0.03091       0.03091
from_dataframe:1000000               1.05764       1.05122
----------------------------------------------------------------
simple_moving_average:100            0.00108       0.00108
simple_moving_average:10000          0.00150       0.00150
simple_moving_average:1000000        0.00839       0.00839
----------------------------------------------------------------
select_and_glue:100                  0.00076       0.00076
select_and_glue:10000                0.00074       0.00074
select_and_glue:1000000              0.00104       0.00104
----------------------------------------------------------------
...
================================================================
```

## Run documentation server locally

Live preview your local changes to the documentation with

```shell
mkdocs serve -f docs/mkdocs.yml
```

## Credits

This project is a collaboration between Google and [Tryolabs](https://tryolabs.com/).
