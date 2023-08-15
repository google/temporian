# Recipes

Short self-contained examples showing how to use Temporian in typical use cases.

## Aggregate events by timestamp

All these recipes show how to combine events based on their timestamps.
For example, to convert events with non-uniform sampling into a time-series with
a fixed sampling interval, combine events with duplicated timestamps,
or aggregate across different indexes.

| Recipe                                                                |
| --------------------------------------------------------------------- |
| [Aggregate events at a fixed interval](aggregate_interval.ipynb)      |
| [Aggregate events from different indexes](aggregate_index.ipynb)      |
| [Unify events with duplicated timestamps](aggregate_duplicated.ipynb) |
