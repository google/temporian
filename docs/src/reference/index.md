# API Reference

This page gives an overview of all of Temporian's public symbols.

Check the index on the left for a more detailed description of any symbol.

## Classes

| Symbol                                        | Description                                                                                                     |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| [`tp.EventSetNode`][temporian.EventSetNode]   | Reference to the input or output of an operator in the compute graph.                                           |
| [`tp.EventSet`][temporian.EventSet]           | Container for actual temporal data.                                                                             |
| [`tp.Schema`][temporian.Schema]               | Description of the data inside an [`EventSetNode`][temporian.EventSetNode] or [`EventSet`][temporian.EventSet]. |
| [`tp.FeatureSchema`][temporian.FeatureSchema] | Description of a feature inside a [`Schema`][temporian.Schema].                                                 |
| [`tp.IndexSchema`][temporian.IndexSchema]     | Description of an index inside a [`Schema`][temporian.Schema].                                                  |

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
| [`tp.from_polars()`][temporian.from_polars] | Converts a Polars DataFrame into an [`EventSet`][temporian.EventSet]. |
| [`tp.to_polars()`][temporian.to_polars]     | Converts an [`EventSet`][temporian.EventSet] to a polars DataFrame.   |
| [`tp.from_csv()`][temporian.from_csv]       | Reads an [`EventSet`][temporian.EventSet] from a CSV file.            |
| [`tp.to_csv()`][temporian.to_csv]           | Saves an [`EventSet`][temporian.EventSet] to a CSV file.              |

## Durations

| Symbols                                                                                                                                                                                                                                                                                                                         | Description                                                                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| [`tp.duration.milliseconds()`][temporian.duration.milliseconds] [`tp.duration.seconds()`][temporian.duration.seconds] [`tp.duration.minutes()`][temporian.duration.minutes] [`tp.duration.hours()`][temporian.duration.hours] [`tp.duration.days()`][temporian.duration.days] [`tp.duration.weeks()`][temporian.duration.weeks] | Convert input value from milliseconds / seconds / minutes / hours / days / weeks to a `Duration` in seconds. |

## Operators

| Symbols                                                                                                    | Description                                                                                                    |
| ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| [`tp.combine()`][temporian.combine]                                                                        | Combines events from [`EventSets`][temporian.EventSet] with different samplings.                               |
| [`tp.glue()`][temporian.glue]                                                                              | Concatenates features from [`EventSets`][temporian.EventSet] with the same sampling.                           |
| [`EventSet.abs()`][temporian.EventSet.abs]                                                                 | Computes the absolute value of the features.                                                                   |
| [`EventSet.add_index()`][temporian.EventSet.add_index]                                                     | Adds indexes to an [`EventSet`][temporian.EventSet].                                                           |
| [`EventSet.arccos()`][temporian.EventSet.arccos]                                                           | Computes the inverse cosine of the features.                                                                   |
| [`EventSet.arcsin()`][temporian.EventSet.arcsin]                                                           | Computes the inverse sine of the features.                                                                     |
| [`EventSet.arctan()`][temporian.EventSet.arctan]                                                           | Computes the inverse tangent of the features.                                                                  |
| [`EventSet.begin()`][temporian.EventSet.begin]                                                             | Generates a single timestamp at the beginning of the input.                                                    |
| [`EventSet.cast()`][temporian.EventSet.cast]                                                               | Casts the dtype of features.                                                                                   |
| [`EventSet.cos()`][temporian.EventSet.cos]                                                                 | Computes the cosine of the features.                                                                           |
| [`EventSet.drop_index()`][temporian.EventSet.drop_index]                                                   | Removes indexes from an [`EventSet`][temporian.EventSet].                                                      |
| [`EventSet.end()`][temporian.EventSet.end]                                                                 | Generates a single timestamp at the end of the input.                                                          |
| [`EventSet.enumerate()`][temporian.EventSet.enumerate]                                                     | Creates an ordinal feature enumerating the events according to their timestamp.                                |
| [`EventSet.equal()`][temporian.EventSet.equal]                                                             | Creates boolean features with event-wise equality to another `EventSet` or to a scalar value.                  |
| [`EventSet.experimental_fast_fourier_transform()`][temporian.EventSet.experimental_fast_fourier_transform] | Applies a Fast Fourier Transform.                                                                              |
| [`EventSet.filter()`][temporian.EventSet.filter]                                                           | Filters out events in an [`EventSet`][temporian.EventSet] for which a condition is false.                      |
| [`EventSet.filter_empty_index()`][temporian.EventSet.filter_empty_index]                                   | Filters out indexes without events.                                                                            |
| [`EventSet.filter_moving_count()`][temporian.EventSet.filter_moving_count]                                 | Skips events such that no more than one event is within a time window of `window_length`.                      |
| [`EventSet.isnan()`][temporian.EventSet.isnan]                                                             | Event-wise boolean that is `True` in the `NaN` positions of the input events. Equivalent to `~evset.notnan()`. |
| [`EventSet.join()`][temporian.EventSet.join]                                                               | Joins [`EventSets`][temporian.EventSet] with different samplings but the same index together.                  |
| [`EventSet.lag()`][temporian.EventSet.lag]                                                                 | Adds a delay to an [`EventSet`][temporian.EventSet]'s timestamps.                                              |
| [`EventSet.leak()`][temporian.EventSet.leak]                                                               | Subtracts a duration from an [`EventSet`][temporian.EventSet]'s timestamps.                                    |
| [`EventSet.log()`][temporian.EventSet.log]                                                                 | Computes natural logarithm of the features                                                                     |
| [`EventSet.map()`][temporian.EventSet.map]                                                                 | Applies a function on each of an [`EventSet`][temporian.EventSet]'s values.                                    |
| [`EventSet.notnan()`][temporian.EventSet.notnan]                                                           | Event-wise boolean that is `True` if the input values are not `NaN`. Equivalent to `~evset.isnan()`.           |
| [`EventSet.prefix()`][temporian.EventSet.prefix]                                                           | Adds a prefix to the names of the features in an [`EventSet`][temporian.EventSet].                             |
| [`EventSet.propagate()`][temporian.EventSet.propagate]                                                     | Propagates feature values over a sub index.                                                                    |
| [`EventSet.rename()`][temporian.EventSet.rename]                                                           | Renames an [`EventSet`][temporian.EventSet]'s features and index.                                              |
| [`EventSet.resample()`][temporian.EventSet.resample]                                                       | Resamples an [`EventSet`][temporian.EventSet] at each timestamp of another [`EventSet`][temporian.EventSet].   |
| [`EventSet.select()`][temporian.EventSet.select]                                                           | Selects a subset of features from an [`EventSet`][temporian.EventSet].                                         |
| [`EventSet.select_index_values()`][temporian.EventSet.select_index_values]                                 | Selects a subset of index values from an [`EventSet`][temporian.EventSet].                                     |
| [`EventSet.set_index()`][temporian.EventSet.set_index]                                                     | Replaces the indexes in an [`EventSet`][temporian.EventSet].                                                   |
| [`EventSet.sin()`][temporian.EventSet.sin]                                                                 | Computes the sine of the features.                                                                             |
| [`EventSet.since_last()`][temporian.EventSet.since_last]                                                   | Computes the amount of time since the last distinct timestamp.                                                 |
| [`EventSet.tan()`][temporian.EventSet.tan]                                                                 | Computes the tangent of the features.                                                                          |
| [`EventSet.tick()`][temporian.EventSet.tick]                                                               | Generates timestamps at regular intervals in the range of a guide.                                             |
| [`EventSet.tick_calendar()`][temporian.EventSet.tick]                                                      | Generates timestamps at the specified calendar date-time events.                                               |
| [`EventSet.timestamps()`][temporian.EventSet.timestamps]                                                   | Creates a feature from the events timestamps (`float64`).                                                      |
| [`EventSet.unique_timestamps()`][temporian.EventSet.unique_timestamps]                                     | Removes events with duplicated timestamps from an [`EventSet`][temporian.EventSet].                            |
| [`EventSet.until_next()`][temporian.EventSet.until_next]                                                   | Duration until the next sampling event.                                                                        |
| [`EventSet.where()`][temporian.EventSet.where]                                                             | Choose events from two possible sources, based on boolean conditions.                                          |

### Calendar operators

| Symbols                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Description                                                                                                                                                          |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`EventSet.calendar_day_of_month()`][temporian.EventSet.calendar_day_of_month] [`EventSet.calendar_day_of_week()`][temporian.EventSet.calendar_day_of_week] [`EventSet.calendar_day_of_year()`][temporian.EventSet.calendar_day_of_year] [`EventSet.calendar_hour()`][temporian.EventSet.calendar_hour] [`EventSet.calendar_iso_week()`][temporian.EventSet.calendar_iso_week] [`EventSet.calendar_minute()`][temporian.EventSet.calendar_minute] [`EventSet.calendar_month()`][temporian.EventSet.calendar_month] [`EventSet.calendar_second()`][temporian.EventSet.calendar_second] [`EventSet.calendar_year()`][temporian.EventSet.calendar_year] | Obtain the day of month / day of week / day of year / hour / ISO week / minute / month / second / year the timestamps in an [`EventSet`][temporian.EventSet] are in. |

### Window operators

| Symbols                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Description                                                                           |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| [`EventSet.simple_moving_average()`][temporian.EventSet.simple_moving_average] [`EventSet.moving_standard_deviation()`][temporian.EventSet.moving_standard_deviation] [`EventSet.cumsum()`][temporian.EventSet.cumsum] [`EventSet.moving_sum()`][temporian.EventSet.moving_sum] [`EventSet.moving_count()`][temporian.EventSet.moving_count] [`EventSet.moving_min()`][temporian.EventSet.moving_min] [`EventSet.moving_max()`][temporian.EventSet.moving_max] [`EventSet.cumprod()`][temporian.EventSet.cumprod] [`EventSet.moving_product()`][temporian.EventSet.moving_product] [`EventSet.moving_quantile()`][temporian.EventSet.moving_quantile] | Compute an operation on the values in a sliding window over an EventSet's timestamps. |

### Python operators

| Symbols                                                                                                   | Description                                                                                                                                                                                      |
| --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `+` (add), `-` (subtract), `\*` (multiply), `/` (divide), `//` (floor divide), `%` (modulo), `**` (power) | Event-wise arithmetic operations between two `EventSets` or with a scalar number. See the corresponding [User Guide section](../user_guide#arithmetic-operators) for more info.                  |
| `!=` (not equal), `>` (greater than), `>=` (greater or equal), `<` (less), `<=` (less or equal)           | Event-wise comparison between `EventSets` or to a scalar number. Note that `==` is not supported, use `EventSet.equal()` instead. See the [User Guide](../user_guide#comparisons) for more info. |
| `&` (and), `\|` (or), `^` (xor)                                                                           | Event-wise logic operators between boolean `EventSets`.                                                                                                                                          |
