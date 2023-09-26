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

| Symbols                                                                                                    | Description                                                                                                  |
| ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| [`tp.combine()`][temporian.combine]                                                                        | Combines events from [`EventSets`][temporian.EventSet] with different samplings.                             |
| [`tp.glue()`][temporian.glue]                                                                              | Concatenates features from [`EventSets`][temporian.EventSet] with the same sampling.                         |
| [`EventSet.add_index()`][temporian.EventSet.add_index]                                                     | Adds indexes to an [`EventSet`][temporian.EventSet].                                                         |
| [`EventSet.begin()`][temporian.EventSet.begin]                                                             | Generates a single timestamp at the beginning of the input.                                                  |
| [`EventSet.cast()`][temporian.EventSet.cast]                                                               | Casts the dtype of features.                                                                                 |
| [`EventSet.drop_index()`][temporian.EventSet.drop_index]                                                   | Removes indexes from an [`EventSet`][temporian.EventSet].                                                    |
| [`EventSet.end()`][temporian.EventSet.end]                                                                 | Generates a single timestamp at the end of the input.                                                        |
| [`EventSet.enumerate()`][temporian.EventSet.enumerate]                                                     | Creates an ordinal feature enumerating the events according to their timestamp.                              |
| [`EventSet.experimental_fast_fourier_transform()`][temporian.EventSet.experimental_fast_fourier_transform] | Apply a Fast Fourier Transform.                                                                              |
| [`EventSet.filter()`][temporian.EventSet.filter]                                                           | Filters out events in an [`EventSet`][temporian.EventSet] for which a condition is false.                    |
| [`EventSet.filter_moving_count()`][temporian.EventSet.filter_moving_count]                                 | Skips events such that no more than one event is within a time window of `window_length`.                    |
| [`EventSet.join()`][temporian.EventSet.join]                                                               | Join [`EventSets`][temporian.EventSet] with different samplings but the same index together.                 |
| [`EventSet.lag()`][temporian.EventSet.lag]                                                                 | Adds a delay to an [`EventSet`][temporian.EventSet]'s timestamps.                                            |
| [`EventSet.leak()`][temporian.EventSet.leak]                                                               | Subtracts a duration from an [`EventSet`][temporian.EventSet]'s timestamps.                                  |
| [`EventSet.prefix()`][temporian.EventSet.prefix]                                                           | Adds a prefix to the names of the features in an [`EventSet`][temporian.EventSet].                           |
| [`EventSet.propagate()`][temporian.EventSet.propagate]                                                     | Propagates feature values over a sub index.                                                                  |
| [`EventSet.rename()`][temporian.EventSet.rename]                                                           | Renames an [`EventSet`][temporian.EventSet]'s features and index.                                            |
| [`EventSet.resample()`][temporian.EventSet.resample]                                                       | Resamples an [`EventSet`][temporian.EventSet] at each timestamp of another [`EventSet`][temporian.EventSet]. |
| [`EventSet.select()`][temporian.EventSet.select]                                                           | Selects a subset of features from an [`EventSet`][temporian.EventSet].                                       |
| [`EventSet.select_index_values()`][temporian.EventSet.select_index_values]                                 | Selects a subset of index values from an [`EventSet`][temporian.EventSet].                                   |
| [`EventSet.set_index()`][temporian.EventSet.set_index]                                                     | Replaces the indexes in an [`EventSet`][temporian.EventSet].                                                 |
| [`EventSet.since_last()`][temporian.EventSet.since_last]                                                   | Computes the amount of time since the last distinct timestamp.                                               |
| [`EventSet.tick()`][temporian.EventSet.tick]                                                               | Generates timestamps at regular intervals in the range of a guide.                                           |
| [`EventSet.timestamps()`][temporian.EventSet.timestamps]                                                   | Creates a feature from the events timestamps (`float64`).                                                    |
| [`EventSet.unique_timestamps()`][temporian.EventSet.unique_timestamps]                                     | Removes events with duplicated timestamps from an [`EventSet`][temporian.EventSet].                          |
| [`EventSet.until_next()`][temporian.EventSet.until_next]                                                   | Duration until the next sampling event.                                                                      |
| [`EventSet.where()`][temporian.EventSet.where]                                                             | Choose events from two possible sources, based on boolean conditions.                                        |

### Binary operators

| Symbols                                                                                                                                                                                                                                           | Description                                                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| [`tp.add()`][temporian.add] [`tp.subtract()`][temporian.subtract] [`tp.multiply()`][temporian.multiply] [`tp.divide()`][temporian.divide] [`tp.floordiv()`][temporian.floordiv] [`tp.modulo()`][temporian.modulo] [`tp.power()`][temporian.power] | Compute an arithmetic binary operation between two [`EventSets`][temporian.EventSet]. Aliases for `+`, `-`, `\*`, `/`, etc. |
| [`tp.equal()`][temporian.equal] [`tp.not_equal()`][temporian.not_equal] [`tp.greater()`][temporian.greater] [`tp.greater_equal()`][temporian.greater_equal] [`tp.less()`][temporian.less] [`tp.less_equal()`][temporian.less_equal]               | Compute a relational binary operator between two [`EventSets`][temporian.EventSet].                                         |
| [`tp.logical_and()`][temporian.logical_and] [`tp.logical_or()`][temporian.logical_or] [`tp.logical_xor()`][temporian.logical_xor]                                                                                                                 | Compute a logical binary operation between two [`EventSets`][temporian.EventSet].                                           |

### Unary operators

| Symbols                                                                                                                                                     | Description                                                      |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| [`tp.abs()`][temporian.abs] [`tp.log()`][temporian.log] [`tp.invert()`][temporian.invert] [`tp.isnan()`][temporian.isnan] [`tp.notnan()`][temporian.notnan] | Compute unary operations on an [`EventSet`][temporian.EventSet]. |

### Calendar operators

| Symbols                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Description                                                                                                                                                          |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`EventSet.calendar_day_of_month()`][temporian.EventSet.calendar_day_of_month] [`EventSet.calendar_day_of_week()`][temporian.EventSet.calendar_day_of_week] [`EventSet.calendar_day_of_year()`][temporian.EventSet.calendar_day_of_year] [`EventSet.calendar_hour()`][temporian.EventSet.calendar_hour] [`EventSet.calendar_iso_week()`][temporian.EventSet.calendar_iso_week] [`EventSet.calendar_minute()`][temporian.EventSet.calendar_minute] [`EventSet.calendar_month()`][temporian.EventSet.calendar_month] [`EventSet.calendar_second()`][temporian.EventSet.calendar_second] [`EventSet.calendar_year()`][temporian.EventSet.calendar_year] | Obtain the day of month / day of week / day of year / hour / ISO week / minute / month / second / year the timestamps in an [`EventSet`][temporian.EventSet] are in. |

### Scalar operators

| Symbols                                                                                                                                                                                                                                                                                                                                             | Description                                                                                     |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| [`tp.add_scalar()`][temporian.add_scalar] [`tp.subtract_scalar()`][temporian.subtract_scalar] [`tp.multiply_scalar()`][temporian.multiply_scalar] [`tp.divide_scalar()`][temporian.divide_scalar] [`tp.floordiv_scalar()`][temporian.floordiv_scalar] [`tp.modulo_scalar()`][temporian.modulo_scalar] [`tp.power_scalar()`][temporian.power_scalar] | Compute an arithmetic operation between an [`EventSet`][temporian.EventSet] and a scalar value. |
| [`tp.equal_scalar()`][temporian.equal_scalar] [`tp.not_equal_scalar()`][temporian.not_equal_scalar] [`tp.greater_equal_scalar()`][temporian.greater_equal_scalar] [`tp.greater_scalar()`][temporian.greater_scalar] [`tp.less_equal_scalar()`][temporian.less_equal_scalar] [`tp.less_scalar()`][temporian.less_scalar]                             | Compute a relational operation between an [`EventSet`][temporian.EventSet] and a scalar value.  |

### Window operators

| Symbols                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Description                                                                           |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| [`EventSet.simple_moving_average()`][temporian.EventSet.simple_moving_average] [`EventSet.moving_standard_deviation()`][temporian.EventSet.moving_standard_deviation] [`EventSet.cumsum()`][temporian.EventSet.cumsum] [`EventSet.moving_sum()`][temporian.EventSet.moving_sum] [`EventSet.moving_count()`][temporian.EventSet.moving_count] [`EventSet.moving_min()`][temporian.EventSet.moving_min] [`EventSet.moving_max()`][temporian.EventSet.moving_max] | Compute an operation on the values in a sliding window over an EventSet's timestamps. |
