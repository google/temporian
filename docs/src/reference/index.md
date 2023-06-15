# API Reference

This page gives an overview of all of Temporian's public symbols.

Check the index on the left for a more detailed description of any symbol.

## Classes

| Symbol                              | Description                                                           |
| ----------------------------------- | --------------------------------------------------------------------- |
| [`tp.Node`][temporian.Node]         | Reference to the input or output of an operator in the compute graph. |
| [`tp.EventSet`][temporian.EventSet] | Container for actual temporal data.                                   |
| [`tp.Schema`][temporian.Schema]     | Description of the data inside a Node or EventSet.                    |

## Functions

| Symbol                                    | Description                                                        |
| ----------------------------------------- | ------------------------------------------------------------------ |
| [`tp.evaluate()`][temporian.evaluate]     | Evaluates Nodes on EventSets.                                      |
| [`tp.plot()`][temporian.plot]             | Plots EventSets.                                                   |
| [`tp.event_set()`][temporian.event_set]   | Creates an EventSet from arrays (list, numpy, pandas).             |
| [`tp.input_node()`][temporian.input_node] | Creates an input node, that can be used to feed data into a graph. |

## Input/output

| Symbol                                      | Description                                   |
| ------------------------------------------- | --------------------------------------------- |
| [`tp.from_pandas()`][temporian.from_pandas] | Converts a Pandas DataFrame into an EventSet. |
| [`tp.to_pandas()`][temporian.to_pandas]     | Converts an EventSet to a pandas DataFrame.   |
| [`tp.from_csv()`][temporian.from_csv]       | Reads an EventSet from a CSV file.            |
| [`tp.to_csv()`][temporian.to_csv]           | Saves an EventSet to a CSV file.              |

## Durations

| Symbols                                                                                                                                                                                                                                                                                                                         | Description                                                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| [`tp.duration.milliseconds()`][temporian.duration.milliseconds] [`tp.duration.seconds()`][temporian.duration.seconds] [`tp.duration.minutes()`][temporian.duration.minutes] [`tp.duration.hours()`][temporian.duration.hours] [`tp.duration.days()`][temporian.duration.days] [`tp.duration.weeks()`][temporian.duration.weeks] | Convert input value from milliseconds / seconds / minutes / hours / days / weeks to a `Duration` in seconds. |

## Operators

| Symbols                                                 | Description                                                        |
| ------------------------------------------------------- | ------------------------------------------------------------------ |
| [`tp.add_index()`][temporian.add_index]                 | Adds one or more features as index in a Node.                      |
| [`tp.begin()`][temporian.begin]                         | Generates a single timestamp at the beginning of the input.        |
| [`tp.cast()`][temporian.cast]                           | Casts the dtype of features.                                       |
| [`tp.drop_index()`][temporian.drop_index]               | Removes one or more index columns from a Node.                     |
| [`tp.end()`][temporian.end]                             | Generates a single timestamp at the end of the input.              |
| [`tp.filter()`][temporian.filter]                       | Filters out events in a node for which a condition is false.       |
| [`tp.glue()`][temporian.glue]                           | Concatenates Nodes with the same sampling.                         |
| [`tp.lag()`][temporian.lag]                             | Adds a delay to a Node's timestamps.                               |
| [`tp.leak()`][temporian.leak]                           | Subtracts a duration from a Node's timestamps.                     |
| [`tp.prefix()`][temporian.prefix]                       | Adds a prefix to the names of the features in a Node.              |
| [`tp.propagate()`][temporian.propagate]                 | Propagates feature values over a sub index.                        |
| [`tp.rename()`][temporian.rename]                       | Renames a Node's features and index.                               |
| [`tp.resample()`][temporian.resample]                   | Resamples a Node at each timestamp of another Node.                |
| [`tp.select()`][temporian.select]                       | Selects a subset of features from a Node.                          |
| [`tp.since_last()`][temporian.since_last]               | Computes the amount of time since the last distinct timestamp.     |
| [`tp.tick()`][temporian.tick]                           | Generates timestamps at regular intervals in the range of a guide. |
| [`tp.unique_timestamps()`][temporian.unique_timestamps] | Removes events with duplicated timestamps from a Node.             |

### Binary operators

| Symbols                                                                                                                                                                                                                                           | Description                                               |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| [`tp.add()`][temporian.add] [`tp.subtract()`][temporian.subtract] [`tp.multiply()`][temporian.multiply] [`tp.divide()`][temporian.divide] [`tp.floordiv()`][temporian.floordiv] [`tp.modulo()`][temporian.modulo] [`tp.power()`][temporian.power] | Compute an arithmetic binary operation between two nodes. |
| [`tp.equal()`][temporian.equal] [`tp.not_equal()`][temporian.not_equal] [`tp.greater()`][temporian.greater] [`tp.greater_equal()`][temporian.greater_equal] [`tp.less()`][temporian.less] [`tp.less_equal()`][temporian.less_equal]               | Compute a relational binary operator between two nodes.   |
| [`tp.logical_and()`][temporian.logical_and] [`tp.logical_or()`][temporian.logical_or] [`tp.logical_xor()`][temporian.logical_xor]                                                                                                                 | Compute a logical binary operation between two Nodes.     |

### Unary operators

| Symbols                                                                                                                                                     | Description                         |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| [`tp.abs()`][temporian.abs] [`tp.log()`][temporian.log] [`tp.invert()`][temporian.invert] [`tp.isnan()`][temporian.isnan] [`tp.notnan()`][temporian.notnan] | Compute unary operations on a Node. |

### Calendar operators

| Symbols                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Description                                                                                                                             |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| [`tp.calendar_day_of_month()`][temporian.calendar_day_of_month] [`tp.calendar_day_of_week()`][temporian.calendar_day_of_week] [`tp.calendar_day_of_year()`][temporian.calendar_day_of_year] [`tp.calendar_hour()`][temporian.calendar_hour] [`tp.calendar_iso_week()`][temporian.calendar_iso_week] [`tp.calendar_minute()`][temporian.calendar_minute] [`tp.calendar_month()`][temporian.calendar_month] [`tp.calendar_second()`][temporian.calendar_second] [`tp.calendar_year()`][temporian.calendar_year] | Obtain the day of month / day of week / day of year / hour / ISO week / minute / month / second / year the timestamps in a Node are in. |

### Scalar operators

| Symbols                                                                                                                                                                                                                                                                                                                                             | Description                                                        |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| [`tp.add_scalar()`][temporian.add_scalar] [`tp.subtract_scalar()`][temporian.subtract_scalar] [`tp.multiply_scalar()`][temporian.multiply_scalar] [`tp.divide_scalar()`][temporian.divide_scalar] [`tp.floordiv_scalar()`][temporian.floordiv_scalar] [`tp.modulo_scalar()`][temporian.modulo_scalar] [`tp.power_scalar()`][temporian.power_scalar] | Compute an arithmetic operation between a Node and a scalar value. |
| [`tp.equal_scalar()`][temporian.equal_scalar] [`tp.not_equal_scalar()`][temporian.not_equal_scalar] [`tp.greater_equal_scalar()`][temporian.greater_equal_scalar] [`tp.greater_scalar()`][temporian.greater_scalar] [`tp.less_equal_scalar()`][temporian.less_equal_scalar] [`tp.less_scalar()`][temporian.less_scalar]                             | Compute a relational operation between a Node and a scalar value.  |

### Window operators

| Symbols                                                                                                                                                                                                                                                                                                                                               | Description                                                                      |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| [`tp.simple_moving_average()`][temporian.simple_moving_average] [`tp.moving_standard_deviation()`][temporian.moving_standard_deviation] [`tp.cumsum()`][temporian.cumsum] [`tp.moving_sum()`][temporian.moving_sum] [`tp.moving_count()`][temporian.moving_count] [`tp.moving_min()`][temporian.moving_min] [`tp.moving_max()`][temporian.moving_max] | Compute an operation on the values in a sliding window over a Node's timestamps. |
