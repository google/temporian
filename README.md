# Temporian

![tests](https://github.com/google/temporian/actions/workflows/test.yaml/badge.svg) ![formatting](https://github.com/google/temporian/actions/workflows/formatting.yaml/badge.svg)


**Temporian** is an open source library designed to streamline the pre-processing of temporal signals, making it faster and more efficient to use them as input features with popular tabular machine learning libraries such as TensorFlow Decision Forests. With Temporian, you can easily perform feature engineering on your temporal data, to improve the accuracy and speed of your forecasting models. Whether you're working with financial data, sensor data, or any other type of time series data, Temporian can help you achieve faster and more accurate results. Best of all, Temporian is completely free and open source, so you can use it however you like, whether for personal or commercial use.



## Usage Example

A minimal end-to-end run looks as follows:

```python
import temporian as t

# Load the data.
event = t.read_event("path/to/data.csv")

# Difficult to explain
schema_event = Event([Feature("sales", int), Feature("costs", int)])

# Create features
sma = t.simple_moving_average(
    input=event,
    window_length=t.day(5),
)

smd = t.simple_moving_standar_deviation(
    input=event,
    window_length=t.day(5),
)

lag = t.lag(
    input=event,
    lag=t.week(1),
)

# Assign features
output_event = t.assign(event, sma)
output_event = t.assign(output_event, smd)
output_event = t.assign(output_event, lag)


# Execute pre processing functions and get results
output_event = t.evaluator.evaluate(
    output_event,
    input_data={
        schema_event: event,
        },
    )

```

>__Warning__: The library is still under construction. This example usage is what we are aiming to build in the short term.

## Supported Features
Temporian currently supports the following features for pre-processing your temporal data:

* **Standard Mean Average:** calculates the average value of each feature over a specified time window.
* **Standard Mean Deviation:** calculates the standard deviation of each feature over a specified time window.
* **Lag:** creates new features by shifting the time series data backward in time by a specified number of time steps.
* **Leak:** creates new features by shifting the time series data forward in time by a specified number of time steps.
* **Arithmetic Operations:** allows you to perform arithmetic operations (such as addition, subtraction, multiplication, and division) on time series data, between different events.



## Requirements for Contributors

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

## Testing for Contributors

Install bazel and buildifier (in Mac we recommend installing bazelisk with brew):

```shell
brew install bazelisk
```

Run all tests with bazel:

```shell
bazel test //...:all
```

>__Note__: You can use the Bazel test flag `--test_output=streamed` to see the test logs in realtime.

## Credits

This project is a collaboration between Google and Tryolabs.
