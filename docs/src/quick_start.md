# 3 minutes to Temporian

This is a _very_ quick introduction to how Temporian works. For a complete tour of its capabilities, please refer to the [User Guide](./user_guide).

## Events and `EventSets`

The most basic unit of data in Temporian is an _event_. An event consists of a timestamp and a set of feature values.

Events are not handled individually. Instead, events are grouped together into **EventSets**, which are Temporian's main class for data handling. You can create an `EventSet` from a pandas DataFrame, a dictionary of NumPy arrays, and more. Here is an example of an `EventSet` containing four events and three features:

```python
evset = tp.EventSet(
	timestamps=["04-02-2023", "06-02-2023", "07-02-2023", "07-02-2023"],
	features={
        "feature_1": [0.5, 0.6, NaN, 0.9],
        "feature_2": ["red", "blue", "red", "blue"],
        "feature_3":  [10, -1, 5, 5],
	},
    index=["feature_2"],
)
```

An `EventSet` can hold one or several _multivariate time sequences_, depending on what its _index_ is. Note that "multivariate" indicates that each event in the time sequence holds several values (one for each feature), and "sequence" indicates that the events are not necessarily sampled at a uniform rate (in which case we would call it a time "series").

If the `EventSet` has no index, it will hold a single time sequence, which means that all events will be considered part of the same group and will interact with each other when operators are applied to the `EventSet`. If the `EventSet` has one (or many) indexes, it will hold one time sequence for each unique value (or unique combination of values) of the indexes, the events will be grouped by the index value they have, and operators applied to the `EventSet` will be applied to each time sequence independently.

## Graph, `Nodes` and Operators

There are two big phases in any Temporian script: graph _definition_ and _evaluation_. This is a common pattern in computing libraries, and it allows us to perform optimizations before the graph is evaluated, to share Temporian programs across different platforms, and more.

Processing operations are performed by _operators_. For example, the [`tp.simple_moving_average()`](./reference/temporian/core/operators/window/simple_moving_average) operator computes the [simple moving average](https://en.wikipedia.org/wiki/Moving_average) of each feature in an `EventSet`. You can find documentation for all available operators [here](./reference/temporian/core/operators). Note that when calling operators you are only defining the processing graph - i.e. you are telling Temporian what operations you want to perform on your data, but those operations are not yet being performed.

Operators are not applied directly to `EventSet`s, but to _nodes_. You can think of a `Node` as the placeholder for an `EventSet` in a preprocessing graph. When applying an operator to a `Node` you are getting back a new `Node` that holds the result of the operation. By combining operators and `Node`s, you can create arbitrarily complex processing graphs.

```python
node = evset.node()

sma_node = tp.simple_moving_average(node, window_length=tp.days(7))
TODO: add more
```

TODO: evaluation part
