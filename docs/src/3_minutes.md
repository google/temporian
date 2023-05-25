# 3 minutes to Temporian

This is a _very_ quick introduction to how Temporian works. For a complete tour of its capabilities, please refer to the [User Guide](/user_guide).

## Events and `EventSets`

The most basic unit of data in Temporian is an **event**. An event consists of a timestamp and a set of feature values.

Events are not handled individually. Instead, events are grouped together into an **[EventSet](/reference/temporian/implementation/numpy/data/event_set)**.

`EventSet`s are the main data structure in Temporian, and represent **[multivariate time sequences](/user_guide/#what-is-temporal-data)**. Note that "multivariate" indicates that each event in the time sequence holds several feature values, and "sequence" indicates that the events are not necessarily sampled at a uniform rate (in which case we would call it a time "series").

You can create an `EventSet` from a pandas DataFrame, NumPy arrays, CSV files, and more. Here is an example of an `EventSet` containing four events and three features:

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

An `EventSet` can hold one or several time sequences, depending on what its **[index](/user_guide/#index-horizontal-and-vertical-operators)** is.

If the `EventSet` has no index, it will hold a single time sequence, which means that all events will be considered part of the same group and will interact with each other when operators are applied to the `EventSet`.

If the `EventSet` has one (or many) indexes, it will hold one time sequence for each unique value (or unique combination of values) of the indexes, the events will be grouped by their index value, and operators applied to the `EventSet` will be applied to each time sequence independently.

## Graph, `Nodes` and Operators

There are two big phases in any Temporian script: graph **definition** and **evaluation**. This is a common pattern in computing libraries, and it allows us to perform optimizations before the graph is evaluated, share Temporian programs across different platforms, and more.

Processing operations are performed by **operators**. For example, the [`tp.simple_moving_average()`](/reference/temporian/core/operators/window/simple_moving_average) operator computes the [simple moving average](https://en.wikipedia.org/wiki/Moving_average) of each feature in an `EventSet`. You can find documentation for all available operators [here](/reference/temporian/core/operators/all_operators).

Note that when calling operators you are only defining the processing graph - i.e. you are telling Temporian what operations you want to perform on your data, but those operations are not yet being performed.

Operators are not applied directly to `EventSet`s, but to **[Nodes](/reference/temporian/core/data/node)**. You can think of a `Node` as the placeholder for an `EventSet` in the processing graph. When applying an operators to `Node`s, you get back new `Node`s that are placeholders for the results of those operations. You can create arbitrarily complex processing graphs by combining operators and nodes.

```python
# Obtain the Node corresponding to the EventSet we created above
source_node = evset.node()

# Apply operators to existing Nodes to generate new Nodes
sma_node = tp.simple_moving_average(source_node, window_length=tp.duration.hours(12))
lagged_sma_node = tp.since_last(sma_node, duration=tp.duration.days(7))
```

<!-- TODO: add image of the generated graph -->

Your preprocessing graph can now be run by calling [`evaluate()`](/reference/temporian/core/data/node/#temporian.core.data.node.Node.evaluate) on any `Node` in the graph, which will perform all necessary operations and return the resulting `EventSet`.

Note that `evaluate()` needs to be passed the `EventSet`s that correspond to the source `Node`s in the graph (since those are not part of the graph definition) and that several `Node`s can be evaluated at the same time by calling [`tp.evaluate()`](/reference/temporian/core/evaluation/#temporian.core.evaluation.evaluate) directly.

```python
lagged_sma_evset = lagged_sma_node.evaluate(evset)
```

🥳 Congratulations! You're all set to write your first pieces of Temporian code.

For a more in-depth look at Temporian's capabilities, please check out the [User Guide](/user_guide) or some of the use cases in the [Tutorials](/tutorials) section.
