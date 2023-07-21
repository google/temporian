# 3 minutes to Temporian

This is a _very_ quick introduction to how Temporian works. For a complete tour of its capabilities, please refer to the [User Guide](../user_guide).

## Events and EventSets

The most basic unit of data in Temporian is an **event**. An event consists of a timestamp and a set of feature values.

Events are not handled individually. Instead, events are grouped together into an **[`EventSet`][temporian.EventSet]**.

[`EventSets`][temporian.EventSet] are the main data structure in Temporian, and represent **[multivariate and multi-index time sequences](../user_guide/#what-is-temporal-data)**. Let's break that down:

- "multivariate" indicates that each event in the time sequence holds several feature values.
- "multi-index" indicates that the events can represent hierarchical data, and be therefore grouped by one or more of their features' values.
- "sequence" indicates that the events are not necessarily sampled at a uniform rate (in which case we would call it a time "series").

You can create an [`EventSet`][temporian.EventSet] from a pandas DataFrame, NumPy arrays, CSV files, and more. Here is an example of an [`EventSet`][temporian.EventSet] containing four events and three features:

```python
>>> evset = tp.event_set(
...     timestamps=["2023-02-04", "2023-02-06", "2023-02-07", "2023-02-07"],
...     features={
...         "feature_1": [0.5, 0.6, np.nan, 0.9],
...         "feature_2": ["red", "blue", "red", "blue"],
...         "feature_3":  [10.0, -1.0, 5.0, 5.0],
...     },
...     indexes=["feature_2"],
... )

```

An [`EventSet`][temporian.EventSet] can hold one or several time sequences, depending on what its **[index](../user_guide/#index-horizontal-and-vertical-operators)** is.

If the [`EventSet`][temporian.EventSet] has no index, it will hold a single multivariate time sequence, which means that all events will be considered part of the same group and will interact with each other when operators are applied to the [`EventSet`][temporian.EventSet].

If the [`EventSet`][temporian.EventSet] has one (or many) indexes, its events will be grouped by their indexes' values, so it will hold one multivariate time sequence for each unique value (or unique combination of values) of its indexes, and most operators applied to the [`EventSet`][temporian.EventSet] will be applied to each time sequence independently.

## Operators

Processing operations are performed by **operators**. For instance, the `tp.simple_moving_average()` operator computes the [simple moving average](https://en.wikipedia.org/wiki/Moving_average) of each feature in an [`EventSet`][temporian.EventSet].

The list of all available operators is available in the [API Reference](./reference/).

```python
>>> # Compute the 2-day simple moving average of the EventSet defined above
>>> sma = tp.simple_moving_average(evset, window_length=tp.duration.days(2))

>>> # Remove index to get a flat EventSet
>>> reindexed = sma.drop_index()

>>> # Subtract feature_1 from feature_3
>>> sub = reindexed["feature_3"] - reindexed["feature_1"]

>>> # Plot the resulting EventSet
>>> sub.plot()

```

## Graph mode

Temporian works in **eager mode** out of the box, which means that when you call an operator on an [`EventSet`][temporian.EventSet] you get back the result of that operation immediately as a new [`EventSet`][temporian.EventSet].

Eager execution is easy to grasp, and fits most small data use cases. However, for big data, **graph mode** allows Temporian to perform optimizations on the computation graph that is defined when operators are applied on [`EventSets`][temporian.EventSet]. Graph mode also enables the serialization of Temporian programs, for later use in other platforms or distributed compute environments.

To learn how graph mode works, check out **[Eager mode vs Graph mode](./user_guide.ipynb#eager-mode-vs-graph-mode)** in the User Guide.

🥳 Congratulations! You're all set to write your first pieces of Temporian code.

For a more in-depth look at Temporian's capabilities, please check out the [User Guide](../user_guide) or some of the use cases in the [Tutorials](../tutorials) section.
