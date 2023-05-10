TODO: rename node to Node?
TODO: add `` to Node and EventSet?

This is a complete guide to Temporian.

If you are are in hurry, we recommend you read the first sections (e.g. util the " Index, horizontal and vertical operators" section) and then look at some of the task-specific [tutorials](https://temporian.readthedocs.io/en/latest/tutorials/).

# Getting started

## ¿What is Temporian?

Temporian is a Python library to feature-engineer temporal data for machine learning models.

For Temporian, temporal data is an **MMTS**. A Multivariate and Multi-index Time Sequence is an extension of multivariate time-series to non-uniform and hierarchically-structured data. A MMTS allows you to represent a time series, as well as other common time data such as transactions, logs, sparse events, asynchronous measurements, or hierarchical records. Temporian unifies all those representations into a powerful and concise framework.

TODO: add plot

## ¿Why Temporian?

With a concise and powerful API, Temporian helps you focus on high level modeling.

To see the benefit of Temporian over general data processing libraries, compare our [original Khipu 2023 Tutorial](TODO: link), which uses pandas to perform feature engineering for the M5 dataset, to the [updated version using Temporian](TODO: link).

Temporian focuses on iterative development in notebooks. You can develop a pre-processing program in a notebook while visualizing results on every step of the way, and then save the program to a `".tem"` file to be consumed in an ML production pipeline with ease.

Finally, Temporian's API is designed to reduce the risk of modeling errors. For instance, Temporian prevents you from inadvertently creating future leakage, that is, using future information to compute signals in the past. This means that you can develop your pipelines with confidence, knowing that your results are accurate and reliable.

## Prerequisites

- Be familiar with Python, and have basic knowledge of Pandas or Numpy.

- Have Temporian [installed](TODO: link).

## Events and EventSets

The most basic unit of data in Temporian is referred to as an _event_. An event consists of a timestamp and a set of feature values.

Here is an example of an event:

```python
"timestamp": 05-02-2023
"feature_1": 0.5
"feature_2": "red"
"feature_3": 10
```

Events are not handled individually. Instead, events are grouped together into **EventSets**. When representing an EventSet, it is convenient to group similar features together and to sort them according to the timestamps in increasing order.

Here is an example of an EventSet containing four events:

```python
"timestamp": [04-02-2023, 06-02-2023, 07-02-2023, 07-02-2023]
"feature_1": [0.5, 0.6, NaN, 0.9]
"feature_2": ["red", "blue", "red", "blue"]
"feature_3":  [10, -1, 5, 5]
```

**Remarks:**

- All the values of a given feature are of the same data type. For instance, `feature_1` is float64 while `feature_2` is a string.
- The value NaN (for _not a number_) indicates that a value is missing.
- Timestamps are not necessarily uniformly sampled.
- The same timestamp can be repeated.

In the next code examples, variables with names like `evset` refer to an EventSet.

You can create an EventSet as follows:

```python
evset = tp.EventSet(
	timestamps = [04-02-2023, 06-02-2023, 07-02-2023, 07-02-2023],
	features = {
        "feature_1": [0.5, 0.6, NaN, 0.9],
        "feature_2": ["red", "blue", "red", "blue"],
        "feature_3":  [10, -1, 5, 5],
	}
)
```

EventSets can be printed.

```python
print(evset)
```

EventSets can be plotted.

```python
evset.plot()
```

**Note:** You'll learn how to create an EventSet using other data sources such as pandas DataFrames later.

Events can carry various meanings. For instance, events can represent **regular measurements**. Suppose an electronic thermometer that generates temperature measurements every minute. This could be an EventSet with one feature called `temperature`. In this scenario, the temperature can change in between two measurements. However, for most practical uses, the most recent measurement will be considered to be the current temperature.

TODO: Temperature plot

Events can also represent the _occurrence_ of sporadic phenomena. Suppose a sales recording system that records client purchases. Each time a client makes a purchase (i.e., each transaction), a new event is created.

TODO: Sales plot

You will see that Temporian is agnostic to the semantics of events, and that often, you will mix together measurements and occurrences. For instance, given the _occurrence_ of sales from the previous example, you can compute daily sales (which is a _measurement_).

## Operators and Processors

Processing operations are performed by _operators_. For instance, the [tp.simple_moving_average()](TODO: link) operator computes the [simple moving average](https://en.wikipedia.org/wiki/Moving_average) of each feature in an EventSet.

Operators are not executed individually, but rather combined to form an operator **Graph**. A graph takes one or multiple EventSets as input and produces one or multiple EventSets as output. Graphs can contain an arbitrary number of operators, which can consume the ouput of other operators as input. You can see a graph as a computation graph where nodes are operators.

TODO: Graph plot

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

# Create an event set compatible with the graph.
a_evset = tp.EventSet(
	timestamps=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
	features={
        "feature_1": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0],
        "feature_2": [54, 23, 53, 12, 12, 32, 23, 12, 2, 4],
	},
    name="a",
)

# Feed the event set to the graph. The result is also an event set.
d_evset = tp.evaluate(d, {a_node: a_evset})

# Print the result.
print(d_evset)
```

The `tp.evaluate` function's signature is `tp.evaluate(<outputs>, <inputs>)`.

The `<outputs>` can be specified as a node, a list of nodes, or a dictionary of names to nodes, and the result of `tp.evaluate` will be of the same type. For example, if `<outputs>` is a list of three nodes, the result will be a list of the three corresponding EventSets.

The `<inputs>` can be specified as a dictionary of nodes to EventSets, a dictionary of names to EventSets, a list of EventSets or a single EventSet, which lets Temporian know the nodes of the graph that each input EventSet corresponds to. In the case of a dictionary of names to EventSets, the names must match the names of nodes in the graph, and in the case of a list or single EventSet, the names of those EventSets must do the same. If specifying the inputs as a dictionary, we could skip passing a name to `a_evset`.

**Remarks:**

- It's important to distinguish between _EventSets_, such as `a_evset`, that contain data, and _nodes_, like `a_node` and `b_node`, that connect operators together and compose the computation graph, but do not contain data.
- No computation is performed during the definition of the graph (i.e., when calling the operator functions). All computation is done during `tp.evaluate`.
- In `tp.evaluate`, the second argument defines a mapping between input nodes and event sets. If all necessary input nodes are not fed, an error will be raised.
- In most cases you will only pass EventSets that correspond to the graph's input nodes, but Temporian also supports passing EventSets to intermediate nodes in the graph. In the example provided, `a_node` is fed, but we could also feed `b_node` and `c_node`. In that case we would not need to feed `a_node`, since no nodes need to be computed from it anymore.

To simplify its usage when the graph contains a single output node, `node.evaluate` is equivalent to `tp.evaluate(node, <inputs>)`.

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

**Warning:** It is more efficient to evaluate multiple output nodes together with `tp.evaluate` than to evaluate them separately with `node_1.evaluate(...)`, `node_2.evaluate(...)`, etc. Only use `node.evaluate` for debugging purposes or when you only have a single output node.

## Implicit input node definition

Previously, we defined the input of the graph `a_node` with `tp.input_node`. This way of listing features manually and their respective data type is cumbersome. If an EventSet is available (i.e., data is available), this step can be changed to use `evset.node()` instead. This is especially useful when creating EventSets from existing data, such as pandas DataFrames or CSV files.

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

# Time units

In Temporian, time is always represented by a float64 value. Users have the freedom to choose the semantic to this value. For example, the time can be the number of nanoseconds since the start of the day, the number of cycles of a process, the number of years since the big bang, or the number of seconds since January 1, 1970, at 00:00:00 UTC, also known as Unix or POSIX time.

To help feature-engineering of dates, Temporian contains a set of _calendar operators_. Those operators are specialized in handling date and datetimes.
For instance, the `tp.calendar_hours` operator returns the hour of the date in the range of 0-23. Calendar operators always assume that the time is Unix time, so applying them on non-Unix timestamps will result in misrepresented features.

```python
>>> a_data = tp.EventSet(
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
>>> a_node = a_data.node()
>>> b_node = tp.glue(a_node, tp.calendar_day_of_week(a_node))
>>> print(b_node.evaluate(a_data))
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

Temporian accepts time inputs in various formats, including integer, float, Python date or datetime, NumPy datetime, and Pandas datetime. Date and datetime objects are internally converted to Unix time in seconds compatible with the calendar operators.

<examples>

Operators can take durations as input arguments. For example, the simple moving average operator takes a `window_length` argument. Temporian exposes several utility functions to help creating those duration arguments when using the Unix timestamps:

```python
# A moving average over 1 day
b = tp.simple_moving_avegage(a, window_length = tp.duration.days(1))

# This line is equivalent to:
b = tp.simple_moving_avegage(a, window_length = 24 * 60 * 60)
```

## Plotting

Data visualization is crucial for gaining insights into data and the system it represents. It also helps in detecting unexpected behavior and issues, making debugging and iterative development easier.

Temporian provides two plotting functions for data visualization: `events.plot(<options>)` and `tp.plot([<list of events>], <option>)`.
The `events.plot()` function is shorter to write and is used for displaying a single event set, while the `tp.plot()` function is used for displaying multiple event sets together. This function is particularly useful when events are indexed (see "Index, horizontal and vertical operators" section) or have different sampling (see "Sampling" section).

Here's an example of using the `events.plot()` function:

```python
events = tp.EventSet(
	timestamps = [1,2,3,4,5],
	features = {
	"feature_1": [0.5, 0.6, 0.4,0.4, 0.9],
	"feature_2": ["red", "blue", "red", "blue"]
}
)
events.plot()
```

By default, the plotting style is selected automatically based on the data. For example, uniformly sampled numerical features (i.e., time-series) are plotted with a continuous line, while non uniformly sampled values are plotted with markers. Those and other behaviors can be controlled using the options

Here's an example of using the `events.plot()` function with options:

```
figure = events.plot(
    style="marker",
    width_px=400,
    min_time=2,
    max_time=10,
    return_fig=True
)
```

By default, the plots are static images. However, interactive plotting can be very powerful. To enable interactive plotting, use interactive=True. Note that interactive plotting requires the "bokeh" Python library to be installed.

```python
!pip install bokeh -q
events.plot(interactive=True)
```

## Feature naming

Each feature is identified by a name, and the list of features is available with the features property.

```python
>>> events = tp.EventSet(
>>> 	timestamps = [1,2,3,4,5],
>>> 	features = {
>>> 	"feature_1": [0.5, 0.6, 0.4,0.4, 0.9],
>>> 	"feature_2": [1.0, 2.0, 3.0, 2.0, 1.0]}
>>> )
>>> node = events.node()
>>> print(node.feature_names)
[“feature_1”, “feature_2”]
```

You can modify feature names using the tp.rename and tp.prefix operators. The `tp.rename` changes the name of a feature (or a set of features) while tp.prefix adds a prefix in front of existing feature names. Note that rename/prefix do not modify the content of the input event. Instead, they return a new event with the modified feature names.

```python
print(tp.rename("renamed_feature_1", node["feature_1"]))
print(tp.rename(["renamed_feature_1", "renamed_feature_2"], node))
```

```python
print(tp.prefix("rename.", node["feature_1"]))
print(tp.prefix("rename.", node))
```

Most operators do not change the feature names. For example, the result of applying the "tp.moving_sum" operator on `feature_1` is called `feature_1`.

```python
>>> tp.moving_sum(node, 1.0).feature_names
[“feature_1”, “feature_2”]
```

It is recommended to use `tp.rename` and `tp.prefix` to organize your data.

```python
a = tp.prefix("sma_7.", tp.simple_moving_average(node, tp.duration.days(7)))
a = tp.prefix("sma_14.", tp.simple_moving_average(node, tp.duration.days(14)))
```

The `tp.glue` operator can be used to concatenate different features into a single event set. The following pattern is commonly used in Temporian programs.

```python
a = tp.glue(
    tp.prefix("sma_7.", tp.simple_moving_average(node, tp.duration.days(7)))
    tp.prefix("sma_14.", tp.simple_moving_average(node, tp.duration.days(14)))
)
```

Some operators can output different feature names. This is notably the case for calendars and some arithmetic operators.

```python
(node["feature_1"] + node["feature_2"]).features
```

## Casting

Temporian is strict feature type (also called dtype). This means that often, you cannot perform operations between features of different types. For example, you cannot subtract a tp.float32 and a tp.float64. Instead, you must manually cast the features to the same type before performing the operation.

```python
a = tp.input_node(features = [("f1", tp.float32), ("f2", tp.float64)])

tp.cast(a["f1"], tp.float64) + a["f2"]
```

Casting is especially useful for reducing memory usage. For example, if a feature only contains values between 0 and 10000, using tp.int32 instead of tp.int64 will reduce memory usage by 2x. This optimization is critical when working with large datasets.

Casting can also be a necessary step before calling operators that only accept certain input data types.

Note that in Python, 1.0 and 1 are respectively float64 and int64 values.

Temporian supports data type casting through the `tp.cast` operator. Destination data types can be specified in three different ways:

Single data type: attempts to convert all input features to the same destination data type.

```python
>>> evset.features
[(“feat_a”, tp.float32), (“feat_b”, tp.float64)]
>>> print(tp.cast(evset, tp.string).features)
[(“feat_a”, tp.string), (“feat_b”, tp.string)]
```

Feature name to data type mapping: converts features (specified by name) to a specific data type.

```python
>>> evset.features
[(“feat_a”, tp.float32), (“feat_b”, tp.float64)]
>>> print(tp.cast(evset, {“feat_a”: tp.string, “feat_b”: tp.int64}).features)
[(“feat_a”, tp.string), (“feat_b”, tp.int64)]
```

Data type to data type mapping: converts all features from a specific data type to another data type.

```python
>>> evset.features
[(“feat_a”, tp.float32), (“feat_b”, tp.float64), (“feat_c”, tp.float64)]
>>> evset_cast = tp.cast(evset, {tp.float32: tp.string, tp.float64: tp.int64})
>>> evset_cast.features
[(“feat_a”, tp.string), (“feat_b”, tp.int64), (“feat_c”, tp.int64)]

Keep in mind that the casting can fail when the graph is evaluated. For instance, attempting to cast “3ba” to a float will result in an error. This errors cannot be caught prior to graph evaluation.
```

## Arithmetic operators

Arithmetic operators are operations between the features of an event set (or between the features of events with similar sampling; see "Sampling" section). Common mathematical and bit operations are supported such as: addition (+), subtraction (-), product (\*), division (/), floor division (//), modulo (%), comparisons (>, >=, <, <=), bitwise (&, |, ~).
These operators can be invoked with python operators or by an explicit call:

```python
evset_add = evset_1 + evset_2

# Or, equivalently
evset_add = tp.add(evset_1, evset_2)
```

Arithmetic operators are applied index-feature-wise and timestamp-wise on events with multiple features.

```python
events = tp.EventSet(
	timestamps = [1, 10],
	features = {
	"f1": [1.0, 11.0],
	"f2": [2.0, 12.0],
"f3": [3.0, 13.0],
"f4": [4.0, 14.0]}
)

events_f12 = events[["f1", "f2"]]
events_f34 = events[["f3", "f4"]]

(events_f12 + events_f34).features
```

**Warning:** The Python equality operator (`==`) does not compute element-wise equality between features of an event set. Instead, the `tp.equal` operator should be used for this purpose.

```python
events = tp.EventSet(
	timestamps = [1, 2],
	features = {
	"f1": [1, 3],
	"f2": [2, 3]}
)
tp.equal(events["f1"], events["f2"]).evaluate(events)
```

Arithmetic operators act feature-wise, i.e. they perform index-feature-wise operations (for each feature in each index key). For operations between EventSets, this implies that the input EventSets must have the same number of features and sampling.

```python
>>> evset_1.feature_names
[“feat_a”, “feat_b”]
>>> evset_2.feature_names
[“feat_c”]
>>> evset_1 + evset_2
ValueError("evset_1 and evset_2 must have same number of features.")
```

```python
>>> evset_1.sampling.index
[“idx_x”, “idx_y”]
>>> evset_2.sampling.index
[“idx_x”, “idx_z”]
>>> evset_1 + evset_2
ValueError("evset_1 and evset_2 must have same sampling.")
```

If you want to apply arithmetic operators on EventSets with different sampling, take a look at the vertical operators section [add link].

For operations involving scalars, operations are applied index-feature-element-wise.

```python
>>> events = tp.EventSet(
>>> 	timestamps = [1,2,3,4],
>>> 	features = {
>>> 	     "feature_1": [0.5, 0.6, 0.4 ,0.4],
>>> 	     "feature_2": [1.0, 2.0, 3.0, 2.0]
>>>        }
>>> )
>>> a_node = events.node()

>>> a_node_add  = a_node + 1.0
>>> print(a_node_add.evaluate(events))
>>> EventSet(
>>> 	timestamps = [1,2,3,4],
>>> 	features = {
>>> 	     "feature_1": [1.5, 1.6, 1.4, 1.4],
>>> 	     "feature_2": [2.0, 3.0, 4.0, 3.0]
>>>        }
>>> )
```

## Sampling

Arithmetic operators, such as the addition operator require for input arguments to have the same timestamps and an index (see next section). The unique combination of timestamps and index is called a _sampling_.

<example>

For example, if nodes `a` and `b` have different samplings, `a["feature_1"] + b["feature_2"]` will fail with the error message:

> > The two operands have different sampling. [details about the samplings]

To use arithmetic operators on event sets with different sampling, one of the event sets needs to be resampled to the sampling of the other event set. Resampling is done with the `tp.resample` operator.

The `tp.resample` operator takes two event sets input called `events` and `sampling`, and returns the resampling of the features of `events` according to the timestamps of `sampling` according to the following rules:

If a timestamp is present in `events` but not in `sampling`, the timestamp is dropped.
If a timestamp is present in both `events` and `sampling`, the timestamp is kept.
If a timestamp is present in `sampling` but not in `events`, a new timestamp is created using the feature values from the _closest anterior_ (not the closest, as that could induce future leakage) timestamp of `events`. This rule is especially useful for events that represent measurements (see "Events and Event Sets" section).

**Note:** Features in `sampling` are ignored.

Here is an example:

```python
a = tp.EventSet(timestamps = [10, 20, 30], features = {"x": ["1.0,2.0,3.0,4.0 ]})
b = tp.EventSet(timestamps = [0, 9, 10, 11, 19, 20, 21])
c = tp.resample(events=a.node(), sampling=b.node())

c.evaluate([a, b])
```

Following the matching between the timestamps of `sampling=b.node()` and `events=a.node()`:

sampling
0
9
10
11
19
20
21
matching event sampling
None
None
10
10
10
20
20
matching event x feature
NaN
(missing)
NaN
(missing)
1
1
1
2
2

If `sampling` contains a timestamp anterior to any `events` timestamp (like 0 and 9 in the example above), the feature of the sampled event will be missing. The representation of a missing value depends on its dtype:

float: NaN
integer: 0
string: "" (empty string)

Back to the example of the `tp.add` operator, `a` and `b` with different sampling can be added in one of the following ways:

```python
a["feature_1"] + tp.resample(b, a) ["feature_2"]
a["feature_1"] + tp.resample(b["feature_2"], a)
# Assuming that "c" is another event set.
tp.resample(a["feature_1"], c) + tp.resample(b["feature_2"], c)
```

`tp.resample` is critical to combine events from different, asynchronized sources. For example, consider a system with two sensors, a thermometer for temperature and a manometer for pressure. The temperature sensor produces measurements every 1 to 10 minutes, while the pressure sensor returns measurements every second. Additionally assume that both sensors are not synchronized. Finally, assume that you need to combine the temperature and pressure measurements with the equation "temperature / pressure".

<Example>

Since the temperature and pressure event set have different sampling, you will need to combine them in one of the following ways.

```python
r = tp.resample(termometer["temperature"], manometer) / manometer["pressure"]
```

The pressure sensor has higher resolution. Therefore, resampling the temperature to the pressure yields higher resolution than resampling the pressure to the temperature.

When handling non uniform sampling it is also common to have a common resampling source.

```python
sampling_source = ... # Uniform timestamps every 10 seconds
r = tp.resample(termometer["temperature"], sampling_source) / tp.resample(manometer["pressure"], sampling_source)
```

Window operators, such as the simple moving average or moving count operators, have an optional `sampling` argument. For example, the signature of the simple moving average operator is `tp.simple_moving_average(events: Node, window_length: Duration, sampling:Optional[Node]=None)`. If `sampling` is not set, the result is sampled the same as the `events` argument. If `sampling` is set, the operator is applied at the timestamps of `sampling`.

```python
b = tp.simple_moving_average(events=a, window_length=10)
c= tp.simple_moving_average(events=a, window_length=10, sampling=d)
```

```python
# The two lines return the same results, but the second line is significantly faster.
c = tp.simple_moving_average(events=a, window_length=10, sampling=b)
c = tp.sample(tp.simple_moving_average(events=a, window_length=10), b)
```

## Index, horizontal and vertical operators

All operators presented so far work on a sequence of related events. For instance, the simple moving average operator computes the average of events within a specific time window. These types of operators are call **horizontal operators**.

It is sometimes desirable for events in an event set not to interact with each other. For example, assume a dataset containing the sum of daily sales of a set of products. The objective is to compute the sum of weekly sales of each product independently. In this scenario, the weekly moving sum should be applied individually to each product. If not, you would compute the weekly sales of all the products together. To compute the weekly sales of individual products, you can define an _index_ on the `product` feature. The moving sum operator will be applied separately to each product.

```python
daily_sales= tp.EventSet(
	timestamps = [...],
	features = {
	"product": [...],
	"sale": [...],
     }
)

a = daily_sales.node()

# Create an index on the "product" feature
b = tp.set_index(a, "product")

# Compute the moving sum of each product individually
c = tp.moving_sum(b, window_length=tp.duration.weeks(1))

b.evaluate({a: daily_sales})
```

Operators that, like the moving sum, are applied independently on each index are called horizontal operators. Operators that modify indexes are called vertical operators. The most important vertical operators are:

`tp.set_index`: Overwrite or add a feature to the existing index.
`tp.drop_index`: Remove an index; optionally convert it into a feature.
`tp.propagate`: Expand an index based on another EventSet’s index.

Note: By default, when creating event sets with `EventSet`, all the events are in a single global index group. Keep in mind only string and integer features can be used as indexes.

We can also have multiple indexes. In the next example, assume our daily sale aggregates are also annotated with `store` data.

```python
daily_sales = tp.EventSet(
	timestamps = [...],
	features = {
	"product": [...],
	"store": [...],
	"price": [...],
	"quantity": [...],
	}
```

Sales records correspond to individual purchases. Instead, let's compute the daily sum of sales for each (product, store) pair.

```python
a = daily_sales.node()
b = tp.glue(a, tp.rename(a["price"] * a["quantity"], “sales”))
b = tp.set_index(a, ["product", "store"] )
# Moving sum computed individually for each (product, store).
c = tp.moving_sum(b["sales"], window_length=tp.duration.weeks(1)) # indexed by ["product", "store"]
```

Now, let's compute the daily sum of sales for each store.

```python
d = tp.set_index(a, "store")
e = tp.moving_sum(d["sales"], window_length=tp.duration.weeks(1))

# Which is equivalent to
d = tp.drop_index(b, "product")
e = tp.moving_sum(d["sales"], window_length=tp.duration.weeks(1))
```

Finally, let's compute the ratio of sales of each pair (product, store) compared to the corresponding (store). Since `c` (daily sales for each product and store) and `e` (daily sales for each store) have different indexes, we cannot use tp.divide (or /) directly - we must first `propagate` `e` to the `[“product”, “store”]` level.

```python
# Copy the content of "e" (indexed by "store") into each (store, product).
f = c / tp.propagate(e, sampling=c, resample=True)

# Or equivalently
f = c / tp.resample(tp.propagate(e, sampling=c), sampling=c)
```

## Future leakage

In supervised learning, [leakage](<https://en.wikipedia.org/wiki/Leakage_(machine_learning)>) is the use of data not available at serving time by a machine learning model. A common example of leakage is _label leakage_, which involves the invalid use of labels in the model input features. Leakage tends to bias model evaluation by making it appear much better than it is in reality. Unfortunately, leakage is often subtle, easy to inject, and challenging to detect.

Another type of leakage is future leakage, where a model uses data before it is available. Future leakage is particularly easy to create and hard to detect, as all feature data is ultimately available to the model; the question is when it is accessed.

To avoid future leakage, Temporian operators are guaranteed to not cause future leakage except for the `tp.leak` operator. This means that it is impossible to inadvertently add future leakage to a Temporian program.

The tp.leak operator can be useful for precomputing labels or evaluating machine learning models. However, its outputs shouldn’t be used as input features. To check programmatically if a node depends on a tp.leak operator, we can use the tp.has_leak function.

```python
>>> a = tp.input_node(features = [("feature_1", tp.float32)])
>>> b = tp.moving_count(a, 1)
>>> c = tp.moving_count(tp.leak(b, 1), 2)

>>> print(""b" has a future leak:", tp.has_leak(b))
False
>>> print(""c" has a future leak:", tp.has_leak(b))
True
```

In this example, `b` does not have a future leak, but `c` does because it depends on `tp.leak`. By using `tp.has_leak`, we can programmatically identify future leakage and modify our code accordingly.

## Accessing event set data

Event set data can be accessed using the `index()` and `feature()` functions. Temporian internally relies on NumPy, which means that the data access functions always return NumPy arrays.

```
# Create an event set
events = tp.EventSet(
	timestamps = [1, 2, 3, 5, 6],
	features = {
	"f1": [0.1, 0.2, 0.3, 1.1, 1.2],
	"f2": ["red", "red", "red", "blue", "blue"],
	},
	indexes=["f2"],
)
events
```

```python
# Access the data for the index "f2=red".
events.index("red")

# Or, equivalently
events.index(("red", ))
```

```python
events.index("red").feature("f1")
```

If an event set does not have an index, `features` can be called directly:

```python
events = tp.EventSet(
	timestamps = [1, 2, 3, 5, 6],
	features = {
	"f1": [0.1, 0.2, 0.3, 1.1, 1.2],
	"f2": ["red", "red", "red", "blue", "blue"],
	},
)
events.feature("f1")

# Or, equivalently
# events.index((,)).feature("f1")
```

## Import and export data

Converting event set data to and from Pandas DataFrames is a straightforward process.

```python
import pandas a pd

df = pd.DataFrame({
"timestamp": [1, 2, 3, 5, 6],
"f1": [0.1, 0.2, 0.3, 1.1, 1.2],
"f2": ["red", "red", "red", "blue", "blue"],
})

df
```

```python
events = tp.from_dataframe(df)
```

```
tp.to_dataframe(events)
```

## Serializing and deserializing of graph

Temporian graphs can be exported and imported to a safe-to-share file with `tp.save` and `tp.load`. In both functions, the input and output nodes are always named.

```python

# Define a Temporian graph
events = tp.EventSet(
	timestamps = [1, 2, 3],
	features = {"f1": [0.1, 0.2, 0.3]},
)
a = events.node()
b = tp.moving_count(a, 1)

# Run the graph
tp.evaluate(b, {a: events})

# Save the graph to file
tp.save(inputs={"input_a": a}, outputs={"output_b": b}, path="/tmp/my_graph.tem")

a.set_name("input_a")
b.set_name("output_b")
...
tp.save(inputs=a, outputs=[b], path="/tmp/my_graph.tem")


# Restore the graph from file
loaded_inputs, loaded_outputs = tp.load(path="/tmp/my_graph.tem")

# Run data on the restored graph
tp.evaluate(loaded_outputs["output_b"], {loaded_inputs["input_a"]: events})
```

## Memory management

tell which operator is copping data, and which one is not
track data usage (different ways)
explain that this is a graphical language. give examples
Measuring memory usage( in eval and in event set)
normalize eventset print

node = evset.node()

node = tp.select(node, …)
node += 3
node = node + 3
node["aze"] = ...

node = tp.source_onde()
node += 1
node += 1
node += 1
a = something(node)

e.evaluate(node:{})

x = ...
x = x....
x = x....
