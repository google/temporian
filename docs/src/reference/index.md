# API Reference

This page gives an overview of all of Temporian's public symbols.

Check the index on the left for a more detailed description of any symbol.

## Classes

| Symbol                                      | Description                                                                                                     |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| [`tp.EventSetNode`][temporian.EventSetNode] | Reference to the input or output of an operator in the compute graph.                                           |
| [`tp.EventSet`][temporian.EventSet]         | Container for actual temporal data.                                                                             |
| [`tp.Schema`][temporian.Schema]             | Description of the data inside an [`EventSetNode`][temporian.EventSetNode] or [`EventSet`][temporian.EventSet]. |

## Functions

| Symbol                                    | Description                                                                                            |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| [`tp.run()`][temporian.run]               | Evaluates [`EventSetNodes`][temporian.EventSetNode] on [`EventSets`][temporian.EventSet].              |
| [`tp.plot()`][temporian.plot]             | Plots [`EventSets`][temporian.EventSet].                                                               |
| [`tp.event_set()`][temporian.event_set]   | Creates an [`EventSet`][temporian.EventSet] from arrays (lists, NumPy arrays, Pandas Series.)          |
| [`tp.input_node()`][temporian.input_node] | Creates an input [`EventSetNode`][temporian.EventSetNode], that can be used to feed data into a graph. |

## Input/output

| Symbol                                      | Description                                                           |
| ------------------------------------------- | --------------------------------------------------------------------- |
| [`tp.from_pandas()`][temporian.from_pandas] | Converts a Pandas DataFrame into an [`EventSet`][temporian.EventSet]. |
| [`tp.to_pandas()`][temporian.to_pandas]     | Converts an [`EventSet`][temporian.EventSet] to a pandas DataFrame.   |
| [`tp.from_csv()`][temporian.from_csv]       | Reads an [`EventSet`][temporian.EventSet] from a CSV file.            |
| [`tp.to_csv()`][temporian.to_csv]           | Saves an [`EventSet`][temporian.EventSet] to a CSV file.              |

## Durations

| Symbols                                                                                                                                                                                                                                                                                                                         | Description                                                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| [`tp.duration.milliseconds()`][temporian.duration.milliseconds] [`tp.duration.seconds()`][temporian.duration.seconds] [`tp.duration.minutes()`][temporian.duration.minutes] [`tp.duration.hours()`][temporian.duration.hours] [`tp.duration.days()`][temporian.duration.days] [`tp.duration.weeks()`][temporian.duration.weeks] | Convert input value from milliseconds / seconds / minutes / hours / days / weeks to a `Duration` in seconds. |

## Operators

| Symbols                                                 | Description                                                                                                                  |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| [`tp.add_index()`][temporian.add_index]                 | Adds indexes to an [`EventSetNode`][temporian.EventSetNode].                                                                 |
| [`EventSet.begin()`][temporian.EventSet.begin]          | Generates a single timestamp at the beginning of the input.                                                                  |
| [`tp.cast()`][temporian.cast]                           | Casts the dtype of features.                                                                                                 |
| [`tp.drop_index()`][temporian.drop_index]               | Removes indexes from an [`EventSetNode`][temporian.EventSetNode].                                                            |
| [`tp.end()`][temporian.end]                             | Generates a single timestamp at the end of the input.                                                                        |
| [`tp.enumerate()`][temporian.enumerate]                 | Creates an ordinal feature enumerating the events according to their timestamp.                                              |
| [`tp.filter()`][temporian.filter]                       | Filters out events in an [`EventSetNode`][temporian.EventSetNode] for which a condition is false.                            |
| [`tp.glue()`][temporian.glue]                           | Concatenates [`EventSetNodes`][temporian.EventSetNode] with the same sampling.                                               |
| [`tp.join()`][temporian.join]                           | Join [`EventSetNodes`][temporian.EventSetNode] with different samplings but the same index together.                         |
| [`tp.lag()`][temporian.lag]                             | Adds a delay to an [`EventSetNode`][temporian.EventSetNode]'s timestamps.                                                    |
| [`tp.leak()`][temporian.leak]                           | Subtracts a duration from an [`EventSetNode`][temporian.EventSetNode]'s timestamps.                                          |
| [`tp.prefix()`][temporian.prefix]                       | Adds a prefix to the names of the features in an [`EventSetNode`][temporian.EventSetNode].                                   |
| [`tp.propagate()`][temporian.propagate]                 | Propagates feature values over a sub index.                                                                                  |
| [`tp.rename()`][temporian.rename]                       | Renames an [`EventSetNode`][temporian.EventSetNode]'s features and index.                                                    |
| [`tp.resample()`][temporian.resample]                   | Resamples an [`EventSetNode`][temporian.EventSetNode] at each timestamp of another [`EventSetNode`][temporian.EventSetNode]. |
| [`tp.select()`][temporian.select]                       | Selects a subset of features from an [`EventSetNode`][temporian.EventSetNode].                                               |
| [`tp.set_index()`][temporian.set_index]                 | Replaces the indexes in an [`EventSetNode`][temporian.EventSetNode].                                                         |
| [`tp.since_last()`][temporian.since_last]               | Computes the amount of time since the last distinct timestamp.                                                               |
| [`tp.tick()`][temporian.tick]                           | Generates timestamps at regular intervals in the range of a guide.                                                           |
| [`tp.timestamps()`][temporian.timestamps]               | Creates a feature from the events timestamps (`float64`).                                                                    |
| [`tp.unique_timestamps()`][temporian.unique_timestamps] | Removes events with duplicated timestamps from an [`EventSetNode`][temporian.EventSetNode].                                  |

### Binary operators

| Symbols                                                                                                                                                                                                                                           | Description                                                                                                                         |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| [`tp.add()`][temporian.add] [`tp.subtract()`][temporian.subtract] [`tp.multiply()`][temporian.multiply] [`tp.divide()`][temporian.divide] [`tp.floordiv()`][temporian.floordiv] [`tp.modulo()`][temporian.modulo] [`tp.power()`][temporian.power] | Compute an arithmetic binary operation between two [`EventSetNodes`][temporian.EventSetNode]. Aliases for `+`, `-`, `\*`, `/`, etc. |
| [`tp.equal()`][temporian.equal] [`tp.not_equal()`][temporian.not_equal] [`tp.greater()`][temporian.greater] [`tp.greater_equal()`][temporian.greater_equal] [`tp.less()`][temporian.less] [`tp.less_equal()`][temporian.less_equal]               | Compute a relational binary operator between two [`EventSetNodes`][temporian.EventSetNode].                                         |
| [`tp.logical_and()`][temporian.logical_and] [`tp.logical_or()`][temporian.logical_or] [`tp.logical_xor()`][temporian.logical_xor]                                                                                                                 | Compute a logical binary operation between two [`EventSetNodes`][temporian.EventSetNode].                                           |

### Unary operators

| Symbols                                                                                                                                                     | Description                                                              |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| [`tp.abs()`][temporian.abs] [`tp.log()`][temporian.log] [`tp.invert()`][temporian.invert] [`tp.isnan()`][temporian.isnan] [`tp.notnan()`][temporian.notnan] | Compute unary operations on an [`EventSetNode`][temporian.EventSetNode]. |

### Calendar operators

| Symbols                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Description                                                                                                                                      |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| [`tp.calendar_day_of_month()`][temporian.calendar_day_of_month] [`tp.calendar_day_of_week()`][temporian.calendar_day_of_week] [`tp.calendar_day_of_year()`][temporian.calendar_day_of_year] [`tp.calendar_hour()`][temporian.calendar_hour] [`tp.calendar_iso_week()`][temporian.calendar_iso_week] [`tp.calendar_minute()`][temporian.calendar_minute] [`tp.calendar_month()`][temporian.calendar_month] [`tp.calendar_second()`][temporian.calendar_second] [`tp.calendar_year()`][temporian.calendar_year] | Obtain the day of month / day of week / day of year / hour / ISO week / minute / month / second / year the timestamps in an EventSetNode are in. |

### Scalar operators

| Symbols                                                                                                                                                                                                                                                                                                                                             | Description                                                                                             |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| [`tp.add_scalar()`][temporian.add_scalar] [`tp.subtract_scalar()`][temporian.subtract_scalar] [`tp.multiply_scalar()`][temporian.multiply_scalar] [`tp.divide_scalar()`][temporian.divide_scalar] [`tp.floordiv_scalar()`][temporian.floordiv_scalar] [`tp.modulo_scalar()`][temporian.modulo_scalar] [`tp.power_scalar()`][temporian.power_scalar] | Compute an arithmetic operation between an [`EventSetNode`][temporian.EventSetNode] and a scalar value. |
| [`tp.equal_scalar()`][temporian.equal_scalar] [`tp.not_equal_scalar()`][temporian.not_equal_scalar] [`tp.greater_equal_scalar()`][temporian.greater_equal_scalar] [`tp.greater_scalar()`][temporian.greater_scalar] [`tp.less_equal_scalar()`][temporian.less_equal_scalar] [`tp.less_scalar()`][temporian.less_scalar]                             | Compute a relational operation between an [`EventSetNode`][temporian.EventSetNode] and a scalar value.  |

### Window operators

| Symbols                                                                                                                                                                                                                                                                                                                                               | Description                                                                               |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| [`tp.simple_moving_average()`][temporian.simple_moving_average] [`tp.moving_standard_deviation()`][temporian.moving_standard_deviation] [`tp.cumsum()`][temporian.cumsum] [`tp.moving_sum()`][temporian.moving_sum] [`tp.moving_count()`][temporian.moving_count] [`tp.moving_min()`][temporian.moving_min] [`tp.moving_max()`][temporian.moving_max] | Compute an operation on the values in a sliding window over an EventSetNode's timestamps. |
