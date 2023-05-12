# Getting Started

This guide will help you get started with `temporian`.

## Installation

Temporian is available on PyPI. To install it, run:

```shell
pip install temporian
```

## A Minimal End-to-End Run

A minimal end-to-end run looks as follows:

```python
import temporian as tp

# Load the data
evset = tp.read_event_set("path/to/data.csv")
node = evset.node()

# Compute features
# TODO: complete with a simple example...

```

## Usage Example

A minimal end-to-end run looks as follows:

```python
import temporian as tp

# Load the data
evset = tp.read_event_set("path/to/data.csv")
node = evset.node()

# Create Simple Moving Average feature
sma_node = tp.simple_moving_average(
    node,
    window_length=tp.day(5),
)

# Create Lag feature
lag_node = tp.lag(
    node,
    lag=tp.week(1),
)

# Glue features
output_node = tp.glue(node, sma_node)
output_node = tp.glue(output_node, lag_node)


# Execute pipeline and get results
output_evset = tp.evaluate(
    output_node,
    input_data={
        node: evset,
    },
)

```
