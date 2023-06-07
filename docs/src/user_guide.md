# User Guide

This is a complete tour of Temporian's capabilities. For a brief introduction to how the library works, please refer to [3 minutes to Temporian](./3_minutes).

## What is temporal data?

For Temporian, temporal data is a **multivariate and multi-index time sequence**: an extension of multivariate time-series to non-uniform and hierarchically-structured data. This allows for representing time series, but also other common time data such as transactions, logs, sparse events, asynchronous measurements, or hierarchical records. Temporian unifies all of those into a powerful and concise framework.

<!-- TODO: add plot -->

## Events and `EventSets`

The most basic unit of data in Temporian is referred to as an _event_. An event consists of a timestamp and a set of feature values.

Here is an example of an event:

```
timestamp: 05-02-2023
feature_1: 0.5
feature_2: "red"
feature_3: 10
```

Events are not handled individually. Instead, events are grouped together into **`EventSets`**. When representing an `EventSet`, it is convenient to group similar features together and to sort them according to the timestamps in increasing order.

Here is an example of an `EventSet` containing four events, each with three features:

```
timestamp: [04-02-2023, 06-02-2023, 07-02-2023, 07-02-2023]
feature_1: [0.5, 0.6, NaN, 0.9]
feature_2: ["red", "blue", "red", "blue"]
feature_3:  [10, -1, 5, 5]
```

**Remarks:**

- All the values of a given feature are of the same data type. For instance, `feature_1` is float64 while `feature_2` is a string.
- The value NaN (for _not a number_) indicates that a value is missing.
- Timestamps are not necessarily uniformly sampled.
- The same timestamp can be repeated.

In the next code examples, variables with names like `evset` refer to an `EventSet`.

You can create an `EventSet` as follows:

```python
evset = tp.EventSet(
	timestamps=[04-02-2023, 06-02-2023, 07-02-2023, 07-02-2023],
	features={
        "feature_1": [0.5, 0.6, NaN, 0.9],
        "feature_2": ["red", "blue", "red", "blue"],
        "feature_3":  [10, -1, 5, 5],
	}
)
```

`EventSets` can be printed.

```python
print(evset)
```

`EventSets` can be plotted.

```python
evset.plot()
```

**Note:** You'll learn how to create an `EventSet` using other data sources such as pandas DataFrames later.

Events can carry various meanings. For instance, events can represent **regular measurements**. Suppose an electronic thermometer that generates temperature measurements every minute. This could be an `EventSet` with one feature called `temperature`. In this scenario, the temperature can change in between two measurements. However, for most practical uses, the most recent measurement will be considered to be the current temperature.

<!-- TODO: Temperature plot -->

Events can also represent the _occurrence_ of sporadic phenomena. Suppose a sales recording system that records client purchases. Each time a client makes a purchase (i.e., each transaction), a new event is created.

<!-- TODO: Sales plot -->

You will see that Temporian is agnostic to the semantics of events, and that often, you will mix together measurements and occurrences. For instance, given the _occurrence_ of sales from the previous example, you can compute daily sales (which is a _measurement_).

## Graph and Operators

Processing operations are performed by **Operators**. For instance, the [`tp.simple_moving_average()`](https://temporian.readthedocs.io/en/latest/reference/temporian/core/operators/window/simple_moving_average/) operator computes the [simple moving average](https://en.wikipedia.org/wiki/Moving_average) of each feature in an `EventSet`.

Operators are not executed individually, but rather combined to form an operator **Graph**. A graph takes one or multiple `EventSets` as input and produces one or multiple `EventSets` as output. Graphs can contain an arbitrary number of operators, which can consume the ouput of other operators as input. You can see a graph as a computation graph where `Nodes` are operators.

<!-- TODO: Graph plot -->

Let's see how to compute the simple moving average of two features `feature_1` and `feature_2` using two different window lengths, and then sum the results:

```python
# Define the input of the graph.
a_node = tp.input_node(
    features=[
        ("feature_1", tp.float64),
        ("feature_2", tp.int64),
    ],
    index=[("feature_3", tp.string)],
    name="a",
)

# Define the operators in the graph.
b_node = tp.simple_moving_average(a_node, window_length=5)
c_node = tp.simple_moving_average(a_node, window_length=10)
d_node = b_node + c_node

# Create an EventSet compatible with the graph.
a_evset = tp.EventSet(
	timestamps=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
	features={
        "feature_1": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0],
        "feature_2": [54, 23, 53, 12, 12, 32, 23, 12, 2, 4],
	},
    name="a",
)

# Feed the EventSet to the graph. The result is also an EventSet.
d_evset = tp.evaluate(d, {a_node: a_evset})

# Print the result.
print(d_evset)
```

The `tp.evaluate` function's signature is `tp.evaluate(<outputs>, <inputs>)`.

The `<outputs>` can be specified as a `Node`, a list of `Nodes`, or a dictionary of names to `Nodes`, and the result of `tp.evaluate` will be of the same type. For example, if `<outputs>` is a list of three `Nodes`, the result will be a list of the three corresponding `EventSets`.

The `<inputs>` can be specified as a dictionary of `Nodes` to `EventSets`, a dictionary of names to `EventSets`, a list of `EventSets` or a single `EventSet`, which lets Temporian know the `Nodes` of the graph that each input `EventSet` corresponds to. In the case of a dictionary of names to `EventSets`, the names must match the names of `Nodes` in the graph, and in the case of a list or single `EventSet`, the names of those `EventSets` must do the same. If specifying the inputs as a dictionary, we could skip passing a name to `a_evset`.

**Remarks:**

- It's important to distinguish between _`EventSets`_, such as `a_evset`, that contain data, and _`Nodes`_, like `a_node` and `b_node`, that connect operators together and compose the computation graph, but do not contain data.
- No computation is performed during the definition of the graph (i.e., when calling the operator functions). All computation is done during `tp.evaluate`.
- In `tp.evaluate`, the second argument defines a mapping between input `Nodes` and `EventSets`. If all necessary input `Nodes` are not fed, an error will be raised.
- In most cases you will only pass `EventSets` that correspond to the graph's input `Nodes`, but Temporian also supports passing `EventSets` to intermediate `Nodes` in the graph. In the example provided, `a_node` is fed, but we could also feed `b_node` and `c_node`. In that case we would not need to feed `a_node`, since no `Nodes` need to be computed from it anymore.

To simplify its usage when the graph contains a single output `Node`, `node.evaluate` is equivalent to `tp.evaluate(node, <inputs>)`.

```python
# All these statements are equivalent.
tp.evaluate(d_node, {a_node: a_evset})
tp.evaluate(d_node, {"a": a_evset})
tp.evaluate(d_node, [a_evset])
tp.evaluate(d_node, a_evset)
d_node.evaluate({a_node: a_evset})
d_node.evaluate({"a": a_evset})
d_node.evaluate([a_evset])
d_node.evaluate(a_evset)
```

**Warning:** It is more efficient to evaluate multiple output `Nodes` together with `tp.evaluate` than to evaluate them separately with `node_1.evaluate(...)`, `node_2.evaluate(...)`, etc. Only use `node.evaluate` for debugging purposes or when you only have a single output `Node`.

## Creating a `Node` from an `EventSet`

Previously, we defined the input of the graph `a_node` with `tp.input_node`. This way of listing features manually and their respective data type is cumbersome.

If an `EventSet` is available (i.e., data is available) this step can be changed to use `evset.node()` instead, which will return a `Node` that is compatible with it. This is especially useful when creating `EventSets` from existing data, such as pandas DataFrames or CSV files.

```python
# Define an EventSet.
a_evset = tp.EventSet(
	timestamps=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
	features={
        "feature_1": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0],
        "feature_2": [54, 23, 53, 12, 12, 32, 23, 12, 2, 4],
	},
    name="a",
)

# Define the input of a graph using an existing EventSet.
# This line is equivalent to the `tp.input_node` line above.
a_node = a_data.node()

# ... Define operators and evaluate the graph as above.
```

## Time units

In Temporian, times are always represented by a float64 value. Users have the freedom to choose the semantic to this value. For example, the time can be the number of nanoseconds since the start of the day, the number of cycles of a process, the number of years since the big bang, or the number of seconds since January 1, 1970, at 00:00:00 UTC, also known as Unix or POSIX time.

To ease the feature engineering of dates, Temporian contains a set of _calendar operators_. Those operators are specialized in creating features from dates and datetimes. For instance, the `tp.calendar_hours` operator returns the hour of the date in the range `0-23`.

Calendar operators require the time in their inputs to be Unix time, so applying them on non-Unix timestamps will result in errors being raised. Temporian can sometimes automatically recognize if input timestamps correspond to Unix time (e.g. when an `EventSet` is created from a pandas DataFrame with a datetime column, or when passing a list of datetime objects as timestamps in `EventSet`'s constructor). If creating `EventSets` manually and passing floats directly to `timestamps`, you need to explicitly specify whether they correspond to Unix times or not via the `is_unix_timestamp` argument.

```python
>>> a_evset = tp.EventSet(
>>>     timestamps=[
>>>         pd.to_datetime("Monday Mar 13 12:00:00 2023", utc=True),
>>>         pd.to_datetime("Tuesday Mar 14 12:00:00 2023", utc=True),
>>>         pd.to_datetime("Friday Mar 17 00:00:01 2023", utc=True),
>>>     ],
>>>     features={
>>>         "feature_1": [1, 2, 3],
>>>         "feature_2": ["a", "b", "c"],
>>>     },
>>> )
>>> a_node = a_evset.node()
>>> b_node = tp.glue(a_node, tp.calendar_day_of_week(a_node))
>>> print(b_node.evaluate(a_evset))

EventSet(
    timestamps=[
        pd.Timestamp("2023-03-13 12:00:00+0000", tz="UTC"),
        pd.Timestamp("2023-03-14 12:00:00+0000", tz="UTC"),
        pd.Timestamp("2023-03-17 00:00:01+0000", tz="UTC"),
    ],
    features={
        "feature_1": [1, 2, 3],
        "feature_2": ["a", "b", "c"],
        "calendar_day_of_week": [0, 1, 4],
    },
)
```

Temporian accepts time inputs in various formats, including integer, float, Python date or datetime, NumPy datetime, and pandas datetime. Date and datetime objects are internally converted to floats as Unix time in seconds, compatible with the calendar operators.

Operators can take _durations_ as input arguments. For example, the simple moving average operator takes a `window_length` argument. Temporian exposes several utility functions to help creating those duration arguments when using Unix timestamps:

```python
# Define a 1-day moving average.
b = tp.simple_moving_avegage(a, window_length = tp.duration.days(1))

# Equivalent.
b = tp.simple_moving_avegage(a, window_length = 24 * 60 * 60)
```

## Plotting

Data visualization is crucial for gaining insights into data and the system it represents. It also helps in detecting unexpected behavior and issues, making debugging and iterative development easier.

Temporian provides two plotting functions for data visualization: `evset.plot(<options>)` and `tp.plot([<list of EventSets>], <option>)`.

The `evset.plot()` function is shorter to write and is used for displaying a single `EventSet`, while the `tp.plot()` function is used for displaying multiple `EventSets` together. This function is particularly useful when `EventSets` are indexed (see [Index, horizontal and vertical operators](#index-horizontal-and-vertical-operators)) or have different samplings (see [Sampling](#sampling)).

Here's an example of using the `evset.plot()` function:

```python
evset = tp.EventSet(
	timestamps=[1, 2, 3, 4, 5],
	features={
        "feature_1": [0.5, 0.6, 0.4, 0.4, 0.9],
        "feature_2": ["red", "blue", "red", "blue", "green"]
    }
)
evset.plot()
```

By default, the plotting style is selected automatically based on the data.

For example, uniformly sampled numerical features (i.e., time series) are plotted with a continuous line, while non-uniformly sampled values are plotted with markers. Those and other behaviors can be controlled via the function's arguments.

Here's an example of using the `evset.plot()` function with options:

```python
figure = evset.plot(
    style="marker",
    width_px=400,
    min_time=2,
    max_time=10,
    return_fig=True,
)
```

The plots are static images by default. However, interactive plotting can be very powerful. To enable interactive plotting, use `interactive=True`. Note that interactive plotting requires the `bokeh` Python library to be installed.

```python
!pip install bokeh -q
events.plot(interactive=True)
```

## Feature naming

Each feature is identified by a name, and the list of features is available through the `features` property of an `EventSet`.

```python
>>> events = tp.EventSet(
>>> 	timestamps=[1,2,3,4,5],
>>> 	features={
>>> 	    "feature_1": [0.5, 0.6, 0.4, 0.4, 0.9],
>>> 	    "feature_2": [1.0, 2.0, 3.0, 2.0, 1.0]}
>>>     )
>>> node = events.node()
>>> print(node.feature_names)

["feature_1", "feature_2"]
```

Most operators do not change the input feature's names.

```python
>>> tp.moving_sum(node, window_length=10).feature_names

["feature_1", "feature_2"]
```

You can modify feature names using the `tp.rename` and `tp.prefix` operators. `tp.rename` changes the name of features, while `tp.prefix` adds a prefix in front of existing feature names. Note that they do not modify the content of the input `Node`, but return a new `Node` with the modified feature names.

```python
# Rename a single feature.
renamed_f1 = tp.rename("renamed_feature_1", node["feature_1"])

# Rename all features.
renamed_node = tp.rename(["renamed_feature_1", "renamed_feature_2"], node)
```

```python
# Prefix a single feature.
prefixed_f1 = tp.prefix("rename.", node["feature_1"])

# Prefix all features.
prefixed_node = tp.prefix("rename.", node)
```

It is recommended to use `tp.rename` and `tp.prefix` to organize your data.

```python
sma_7_node = tp.prefix("sma_7.", tp.simple_moving_average(node, tp.duration.days(7)))
sma_14_node = tp.prefix("sma_14.", tp.simple_moving_average(node, tp.duration.days(14)))
```

The `tp.glue` operator can be used to concatenate different features into a single `Node`. The following pattern is commonly used in Temporian programs.

```python
result = tp.glue(
    tp.prefix("sma_7.", tp.simple_moving_average(node, tp.duration.days(7))),
    tp.prefix("sma_14.", tp.simple_moving_average(node, tp.duration.days(14))),
)
```

Note however that some operators do output new feature names not present in their inputs. This is notably the case for calendar operators and some arithmetic operators.

```python
>>> print((node["feature_1"] + node["feature_2"]).feature_names)

["add_feature_1_feature_2"]

>>> print(tp.calendar_month(node).feature_names)

["calendar_month"]
```

## Casting

Temporian is strict on feature data types (also called dtype). This means that often, you cannot perform operations between features of different types. For example, you cannot subtract a `tp.float32` and a `tp.float64`. Instead, you must manually cast the features to the same type before performing the operation.

```python
node = tp.input_node(features=[("f1", tp.float32), ("f2", tp.float64)])

added = tp.cast(node["f1"], tp.float64) + node["f2"]
```

Casting is especially useful to reduce memory usage. For example, if a feature only contains values between 0 and 10000, using `tp.int32` instead of `tp.int64` will halve memory usage. These optimizations are critical when working with large datasets.

Casting can also be a necessary step before calling operators that only accept certain input data types.

Note that in Python, 1.0 and 1 are respectively `float64` and `int64` values.

Temporian supports data type casting through the `tp.cast` operator. Destination data types can be specified in three different ways:

1. Single data type: converts all input features to the same destination data type.

   ```python
   >>> node.features

   [("feat_a", tp.float32), ("feat_b", tp.float64)]

   >>> print(tp.cast(node, tp.string).features)

   [("feat_a", tp.string), ("feat_b", tp.string)]
   ```

2. Feature name to data type mapping: converts each feature (specified by name) to a specific data type.

   ```python
   >>> node.features

   [("feat_a", tp.float32), ("feat_b", tp.float64)]

   >>> print(tp.cast(node, {"feat_a": tp.string, "feat_b": tp.int64}).features)

   [("feat_a", tp.string), ("feat_b", tp.int64)]
   ```

3. Data type to data type mapping: converts all features of a specific data type to another data type.

   ```python
   >>> node.features

   [("feat_a", tp.float32), ("feat_b", tp.float64), ("feat_c", tp.float64)]

   >>> print(tp.cast(node, {tp.float32: tp.string, tp.float64: tp.int64}))

   [("feat_a", tp.string), ("feat_b", tp.int64), ("feat_c", tp.int64)]
   ```

Keep in mind that the casting can fail when the graph is evaluated. For instance, attempting to cast `"word"` to `tp.float64` will result in an error. These errors cannot be caught prior to graph evaluation.

## Arithmetic operators

Arithmetic operators are operations between the features of an `EventSet` (or between the features of events with the same sampling. Common mathematical and bit operations are supported, such as addition (+), subtraction (-), product (\*), division (/), floor division (//), modulo (%), comparisons (>, >=, <, <=), and bitwise operators (&, |, ~).

These operators can be invoked with python operators or by an explicit call:

```python
evset_added = evset_1 + evset_2

# Equivalent.
evset_added = tp.add(evset_1, evset_2)
```

Arithmetic operators are applied index-feature-wise and timestamp-wise on events with multiple features.

```python
>>> evset = tp.EventSet(
>>>     timestamps=[1, 10],
>>>     features={
>>>         "f1": [1.0, 11.0],
>>>         "f2": [2.0, 12.0],
>>>         "f3": [3.0, 13.0],
>>>         "f4": [4.0, 14.0],
>>>     },
>>> )
>>> node = evset.node()

>>> node_f12 = node[["f1", "f2"]]
>>> node_f34 = node[["f3", "f4"]]

>>> node_added = (node_f12 + node_f34)
>>> evset_added = node_added.evaluate(evset)
>>> print(evset_added)

EventSet(
	timestamps=[1, 10],
	features={
	    "add_f1_f3": [4.0, 24.0],
	    "add_f2_f4": [6.0, 26.0],
    },
)
```

**Warning:** The Python equality operator (`==`) does not compute element-wise equality between features of an `EventSet`. Use the `tp.equal` operator instead.

```python
evset = tp.EventSet(
	timestamps=[1, 2],
	features={
	"f1": [1, 3],
	"f2": [2, 3]}
)
node = evset.node()

tp.equal(node["f1"], node["f2"]).evaluate(events)
```

Arithmetic operators act feature-wise, i.e. they perform index-feature-wise operations (for each feature in each index key). For operations between `EventSets`, this implies that the input `EventSets` must have the same number of features and sampling.

```python
>>> evset_1.feature_names

["feat_a", "feat_b"]

>>> evset_2.feature_names

["feat_c"]

>>> evset_1 + evset_2

ValueError("evset_1 and evset_2 must have same number of features.")
```

```python
>>> evset_1.sampling.index

["idx_x", "idx_y"]

>>> evset_2.sampling.index

["idx_x", "idx_z"]

>>> evset_1 + evset_2

ValueError("evset_1 and evset_2 must have same sampling.")
```

If you want to apply arithmetic operators on `EventSets` with different samplings, take a look at [Vertical operators](#index-horizontal-and-vertical-operators).

For operations involving scalars, operations are applied index-feature-element-wise.

```python
>>> evset = tp.EventSet(
>>> 	timestamps=[1,2,3,4],
>>> 	features={
>>> 	     "feature_1": [0.5, 0.6, 0.4 ,0.4],
>>> 	     "feature_2": [1.0, 2.0, 3.0, 2.0],
>>>     },
>>> )
>>> node = evset.node()

>>> node_added  = node + 1.0
>>> print(node_added.evaluate(evset))

EventSet(
	timestamps=[1, 2, 3, 4],
	features={
	    "feature_1": [1.5, 1.6, 1.4, 1.4],
	    "feature_2": [2.0, 3.0, 4.0, 3.0],
    },
)
```

## Sampling

Arithmetic operators, such as `tp.add`, require their input arguments to have the same timestamps and [Index](#index-horizontal-and-vertical-operators). The unique combination of timestamps and index is called a _sampling_.

<!-- TODO: example -->

For example, if `Nodes` `a` and `b` have different samplings, `a["feature_1"] + b["feature_2"]` will fail.

To use arithmetic operators on `EventSets` with different samplings, one of the `EventSets` needs to be resampled to the sampling of the other `EventSet`. Resampling is done with the `tp.resample` operator.

The `tp.resample` operator takes two `EventSets` called `input` and `sampling`, and returns the resampling of the features of `input` according to the timestamps of `sampling` according to the following rules:

If a timestamp is present in `input` but not in `sampling`, the timestamp is dropped.
If a timestamp is present in both `input` and `sampling`, the timestamp is kept.
If a timestamp is present in `sampling` but not in `input`, a new timestamp is created using the feature values from the _closest anterior_ (not the closest, as that could induce future leakage) timestamp of `input`. This rule is especially useful for events that represent measurements (see [Events and `EventSets`](#events-and-eventsets)).

**Note:** Features in `sampling` are ignored. This also happens in some other operators that take a `sampling` argument of type `Node` - it indicates that only the sampling (a.k.a. the index and timestamps) of that `Node` are being used by that operator.

Given this example:

```python
evset = tp.EventSet(
    timestamps=[10, 20, 30],
    features={
        "x": [1.0, 2.0, 3.0],
    },
)
sampling = tp.EventSet(
    timestamps=[0, 9, 10, 11, 19, 20, 21],
)

resampled = tp.resample(input=evset.node(), sampling=sampling.node())

resampled.evaluate([evset, sampling])
```

The following would be the matching between the timestamps of `sampling` and `input`:

| `sampling` timestamp         | 0   | 9   | 10  | 11  | 19  | 20  | 21  |
| ---------------------------- | --- | --- | --- | --- | --- | --- | --- |
| matching `input` timestamp   | -   | -   | 10  | 10  | 10  | 20  | 20  |
| matching `"x"` feature value | NaN | NaN | 1   | 1   | 1   | 2   | 2   |

If `sampling` contains a timestamp anterior to any timestamp in the `input` (like 0 and 9 in the example above), the feature of the sampled event will be missing. The representation of a missing value depends on its dtype:

float: `NaN`
integer: `0`
string: `""`

Back to the example of the `tp.add` operator, `a` and `b` with different sampling can be added in one of the following ways:

```python
a["feature_1"] + tp.resample(b, a)["feature_2"]
a["feature_1"] + tp.resample(b["feature_2"], a)

# Assuming that `c` is another EventSet.
tp.resample(a["feature_1"], c) + tp.resample(b["feature_2"], c)
```

`tp.resample` is critical to combine events from different, non-synchronized sources. For example, consider a system with two sensors, a thermometer for temperature and a manometer for pressure. The temperature sensor produces measurements every 1 to 10 minutes, while the pressure sensor returns measurements every second. Additionally assume that both sensors are not synchronized. Finally, assume that you need to combine the temperature and pressure measurements with the equation `temperature / pressure`.

<!-- TODO: image -->

Since the temperature and pressure `EventSets` have different sampling, you will need to resample one of them. The pressure sensor has higher resolution. Therefore, resampling the temperature to the pressure yields higher resolution than resampling the pressure to the temperature.

```python
r = tp.resample(termometer["temperature"], manometer) / manometer["pressure"]
```

When handling non-uniform timestamps it is also common to have a common resampling source.

```python
sampling_source = ... # Uniform timestamps every 10 seconds.
r = tp.resample(termometer["temperature"], sampling_source) / tp.resample(manometer["pressure"], sampling_source)
```

Moving window operators, such as the `tp.simple_moving_average` or `tp.moving_count` operators, have an optional `sampling` argument. For example, the signature of the simple moving average operator is `tp.simple_moving_average(input: Node, window_length: Duration, sampling: Optional[Node] = None)`. If `sampling` is not set, the result will maintain the sampling of the `input` argument. If `sampling` is set, the moving window will be sampled at each timestamp of `sampling` instead, and the result will have those new ones.

```python
b = tp.simple_moving_average(input=a, window_length=10)
c = tp.simple_moving_average(input=a, window_length=10, sampling=d)
```

Note that if planning to resample the result of a moving window operator, passing the `sampling` argument is both more efficient and more accurate than calling `tp.resample` on the result.

## Index, horizontal and vertical operators

All operators presented so far work on a sequence of related events. For instance, the simple moving average operator computes the average of events within a specific time window. These types of operators are called _horizontal operators_.

It is sometimes desirable for events in an `EventSet` not to interact with each other. For example, assume a dataset containing the sum of daily sales of a set of products. The objective is to compute the sum of weekly sales of each product independently. In this scenario, the weekly moving sum should be applied individually to each product. If not, you would compute the weekly sales of all the products together.

To compute the weekly sales of individual products, you can define the `product` feature as the `EventSet`'s _index_. The moving sum operator will then be applied independently to the events corresponding to each product.

```python
daily_sales = tp.EventSet(
	timestamps=[...],
	features={
        "product": [...],
        "sale": [...],
    },
)

a = daily_sales.node()

# Set the "product" feature as the index.
b = tp.add_index(a, "product")

# Compute the moving sum of each product individually.
c = tp.moving_sum(b, window_length=tp.duration.weeks(1))

c.evaluate({a: daily_sales})
```

Horizontal operators can be understood as operators that are applied independently on each index.

Operators that modify an `EventSet`'s index are called _vertical operators_. The most important vertical operators are:

- `tp.add_index`: Add features to the index.
- `tp.drop_index`: Remove features from the index, optionally keeping them as features.
- `tp.set_index`: Changes the index.
- `tp.propagate`: Expand an index based on another `EventSet`’s index.

By default `EventSets` are _flat_, which means they have no index, and therefore all events are in a single global index group.

Also, keep in mind that only string and integer features can be used as indexes.

`EventSets` can have multiple features as index. In the next example, assume our daily sale aggregates are also annotated with `store` data.

```python
daily_sales = tp.EventSet(
	timestamps=[...],
	features={
        "product": [...],
        "store": [...],
        "price": [...],
        "quantity": [...],
	}
)
```

Let's compute the daily sum of sales for each `(product, store)` pair.

```python
a = daily_sales.node()
b = tp.glue(
    a,
    tp.rename(
        a["price"] * a["quantity"],
        "sales"
    ),
)
b = tp.add_index(b, ["product", "store"] )

# Moving sum computed individually for each (product, store).
c = tp.moving_sum(b["sales"], window_length=tp.duration.weeks(1))
```

Now, let's compute the daily sum of sales for each store.

```python
d = tp.add_index(a, "store")
e = tp.moving_sum(d["sales"], window_length=tp.duration.weeks(1))

# Which is equivalent to
d = tp.drop_index(b, "product")
e = tp.moving_sum(d["sales"], window_length=tp.duration.weeks(1))
```

Finally, let's compute the ratio of sales of each `(product, store)` pair compared to the corresponding `store`.

Since `c` (daily sales for each product and store) and `e` (daily sales for each store) have different indexes, we cannot use `tp.divide` (or `/`) directly - we must first `propagate` `e` to the `["product", "store"]` index.

```python
# Copy the content of e (indexed by (store)) into each (store, product).
f = c / tp.propagate(e, sampling=c, resample=True)

# Equivalent.
f = c / tp.resample(
    tp.propagate(e, sampling=c),
    sampling=c,
)
```

The `tp.propagate` operator expands the index of its `input` (`e` in this case) to match the index of its `sampling` by copying the content of `input` into each corresponding index group of `sampling`. Note that the features in `sampling`'s index must be a superset of the ones in `input`'s index.

## Future leakage

In supervised learning, [leakage](<https://en.wikipedia.org/wiki/Leakage_(machine_learning)>) is the use of data not available at serving time by a machine learning model. A common example of leakage is _label leakage_, which involves the invalid use of labels in the model input features. Leakage tends to bias model evaluation by making it appear much better than it is in reality. Unfortunately, leakage is often subtle, easy to inject, and challenging to detect.

Another type of leakage is future leakage, where a model uses data before it is available. Future leakage is particularly easy to create, as all feature data is ultimately available to the model, the problem being it being accessed at the wrong time.

To avoid future leakage, Temporian operators are guaranteed to not cause future leakage, except for the `tp.leak` operator. This means that it is impossible to inadvertently add future leakage to a Temporian program.

`tp.leak` can be useful for precomputing labels or evaluating machine learning models. However, its outputs shouldn’t be used as input features. To check programmatically if a `Node` depends on `tp.leak`, we can use the `tp.has_leak` function.

```python
>>> a = tp.input_node(features=[("feature_1", tp.float32)])
>>> b = tp.moving_count(a, 1)
>>> c = tp.moving_count(tp.leak(b, 1), 2)

>>> print(tp.has_leak(b))
False

>>> print(tp.has_leak(c))
True
```

In this example, `b` does not have a future leak, but `c` does because it depends on `tp.leak`. By using `tp.has_leak`, we can programmatically identify future leakage and modify our code accordingly.

## Accessing `EventSet` data

`EventSet` data can be accessed using the `index()` and `feature()` functions. Temporian internally relies on NumPy, which means that the data access functions always return NumPy arrays.

```python
evset = tp.EventSet(
	timestamps=[1, 2, 3, 5, 6],
	features={
        "f1": [0.1, 0.2, 0.3, 1.1, 1.2],
        "f2": ["red", "red", "red", "blue", "blue"],
	},
	index=["f2"],
)

# Access the data for the index `f2=red`.
evset.index("red")

# Equivalent.
evset.index(("red", ))

# Access the data for the index `f2=red` and feature `f1`.
evset.index("red").feature("f1")
```

If an `EventSet` does not have an index, `feature` can be called directly:

```python
evset = tp.EventSet(
	timestamps=[1, 2, 3, 5, 6],
	features={
        "f1": [0.1, 0.2, 0.3, 1.1, 1.2],
        "f2": ["red", "red", "red", "blue", "blue"],
	},
)
evset.feature("f1")
```

## Import and export data

`EventSets` can be read from and saved to disk via the `tp.read_event_set` and `tp.save_event_set` functions.

```python
# Read EventSet from .csv file.
evset = tp.read_event_set(
    path="path/to/file.csv",
    timestamp_column="timestamp",
    index_names=["product_id"],
)

# Save EventSet to .csv file.
tp.save_event_set(evset, path="path/to/file.csv")
```

Converting `EventSet` data to and from pandas DataFrames is also easily done via `EventSet.to_dataframe` and `tp.pd_dataframe_to_event_set`.

```python
import pandas as pd

df = pd.DataFrame({
    "timestamp": [1, 2, 3, 5, 6],
    "f1": [0.1, 0.2, 0.3, 1.1, 1.2],
    "f2": ["red", "red", "red", "blue", "blue"],
})

# Create EventSet from DataFrame.
evset = tp.pd_dataframe_to_event_set(df)

# Convert EventSet to DataFrame.
df = evset.to_dataframe()
```

## Serialization and deserialization of a graph

Temporian graphs can be exported and imported to a safe-to-share file with `tp.save` and `tp.load`. In both functions input and output `Nodes` need to be named, or be assigned a name by passing them as a dictionary.

```python
# Define a graph.
evset = tp.EventSet(
	timestamps=[1, 2, 3],
	features={"f1": [0.1, 0.2, 0.3]},
)
a = evset.node()
b = tp.moving_count(a, 1)

# Save the graph.
tp.save(inputs={"input_a": a}, outputs={"output_b": b}, path="/tmp/my_graph.tem")

# Equivalent.
a.name = "input_a"
b.name = "output_b"
tp.save(inputs=a, outputs=[b], path="/tmp/my_graph.tem")

# Load the graph.
loaded_inputs, loaded_outputs = tp.load(path="/tmp/my_graph.tem")

# Run data on the restored graph.
tp.evaluate(loaded_outputs["output_b"], {loaded_inputs["input_a"]: evset})
```
