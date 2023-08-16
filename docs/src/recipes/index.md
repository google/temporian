# Recipes

Short self-contained examples showing how to use Temporian in typical use cases.

## Aggregate events by timestamp

All these recipes show how to combine events based on their timestamps.

For example, to convert events with non-uniform sampling into a time-series with
a fixed sampling interval, combine events with duplicated timestamps,
or grouping together events from different indexes.

| Recipe                                                                |
| --------------------------------------------------------------------- |
| [Aggregate events at a fixed interval](aggregate_interval.ipynb)      |
| [Aggregate events from different indexes](aggregate_index.ipynb)      |
| [Unify events with duplicated timestamps](aggregate_duplicated.ipynb) |

## Split EventSets

These recipes show how to split an `EventSet` into multiple subsets.

For example, to create train/validation/test splits for machine learning
applications, based on timestamps or number of samples.

| Recipe                                                    |
| --------------------------------------------------------- |
| [Split data at a given timestamp](split_timestamp.ipynb)  |
| [Split data by fraction of samples](split_fraction.ipynb) |
