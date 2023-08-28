"""Utilities to import/export Beam-Event-Set from/to dataset containers."""

from typing import Iterable, Dict, Any, Tuple, Union, Optional, List, Iterator

import numpy as np
import apache_beam as beam

from temporian.core.data.node import Schema
from temporian.core.data.dtype import DType, tp_dtype_to_py_type
from temporian.implementation.numpy.data.event_set import tp_dtype_to_np_dtype

# Remark: We use tuples instead of dataclasses or named tuples as it seems
# to be the most efficient solution for beam.

# All Temporian computation is done on PCollections of IndexValue.
# A IndexValue contains the timestamps and values of a single feature and is
# indexed with a Temporian index (if any is used in the computation) AND a
# feature index.
#
# An important implication is that timestamps are effectively repeated for
# each feature.

# Temporian index in Beam.
#
# In the numpy backend, index are represented as numpy primitives. However,
# Beam does not support numpy primitive as index. Therefore, all index are
# converted to python primitive of type "BeamIndex".
BeamIndex = Union[int, float, str, bytes, bool]

# Temporian index or Feature index in Beam.
#
# The different features of an EventSets are handled as different item or a
# Beam's PCollection. Each such item, is a IndexValue (see later), and is
# attached to a feature by the index (integer) of this feature in the schema.
#
# An EventSets *without* features is represented as a single such item with a
# feature index of `-1`.
BeamIndexOrFeature = BeamIndex

# A single timestamp value.
Timestamp = np.float64

# A single event / row during the conversion from dict of key/value to internal
# the Temporian beam format for EventSets. In a StructuredRow, index and
# features are indexed by integer instead of string keys.
#
# Contains: the index, the timestamp, and the features.
# The indexes and features are ordered according to a Schema.
# Note the double 2-items tuple (instead of a one 3-items tuple) to facilitate
# Beam operations.
StructuredRow = Tuple[
    Tuple[BeamIndex, ...], Tuple[Timestamp, Tuple[np.generic, ...]]
]

# Unit of data for an EventSet used by all the operators implementations.
#
# Contains: the index+feature_idx, timestamps, and feature values.
# The feature value can be None for EventSet without features.
IndexValue = Tuple[
    Tuple[BeamIndexOrFeature, ...], Tuple[np.ndarray, Optional[np.ndarray]]
]

# From the point of view of the user, a "Beam EventSet" is a PCollection of
# IndexValue.
PEventSet = beam.PCollection[IndexValue]


def _cast_feature_value(value: Any, dtype: DType) -> np.generic:
    """Convert a user feature value to the internal representation."""

    np_type = tp_dtype_to_np_dtype(dtype)
    return np_type(value)


def _cast_index_value(value: Any, dtype: DType) -> BeamIndex:
    """Convert a user index value to the internal representation."""

    return _cast_feature_value(value, dtype).item()


def _reindex_by_integer(
    row: Dict[str, Any], schema: Schema, timestamp_key: str
) -> StructuredRow:
    """Transforms a dict of key:value to a StructuredRow.

    In essence, this function replaces the string index feature and index values
    with a integer index (based on a schema).

    Example:
        row = {"timestamps": 10, "f1": 11, "f2": 12, "i1": 13}
        schema: features = [f1, f2], indexes = [i1]
        timestamp_key: timestamps

        Output
            (13, ), (10, (11, 12))

    This function is used during the conversion of key:value features feed by
    the user into PEventSet, the working format used by Temporian.
    """

    index_values = [
        _cast_index_value(row[index.name], index.dtype)
        for index in schema.indexes
    ]
    feature_values = [
        _cast_feature_value(row[feature.name], feature.dtype)
        for feature in schema.features
    ]
    timestamp = np.float64(row[timestamp_key])
    return tuple(index_values), (timestamp, tuple(feature_values))


class _MergeTimestampsSplitFeatures(beam.DoFn):
    """Aggregates + split StructuredRows into IndexValue.

    This function aggregates together all the timestamps+values having the same
    feature+index, and split then by features.

    Example:
        item
            (20, ), (100, (11, 12))
            (20, ), (101, (13, 14))
            (21, ), (102, (15, 16))

        Output
            (20, 0), ( (100, 101), (11, 13))
            (20, 1), ( (100, 101), (12, 14))
            (21, 0), ( (102,), (15,))
            (21, 1), ( (102,), (16,))

    This function is used during the conversion of key:value features feed by
    the user into PEventSet, the working format used by Temporian.
    """

    def __init__(self, num_features: int):
        self._num_features = num_features

    def process(
        self,
        item: Tuple[
            Tuple[BeamIndex, ...],
            Iterable[Tuple[Timestamp, Tuple[np.generic, ...]]],
        ],
    ) -> Iterable[IndexValue]:
        index, feat_and_ts = item
        timestamps = np.array([v[0] for v in feat_and_ts], dtype=np.float64)
        for feature_idx in range(self._num_features):
            values = np.array([v[1][feature_idx] for v in feat_and_ts])
            yield index + (feature_idx,), (timestamps, values)


def _event_set_dict_to_event_set(
    input: Dict[str, Any], schema: Schema, timestamp_key: str
) -> Iterator[IndexValue]:
    """Converts a `indexedEvents` event-set into an internal event-set.

    Example:
        Input
            Schema
                features=[("f1", DType.INT64), ("f2", DType.STRING)]
                indexes=[("i1", DType.INT64), ("i2", DType.STRING)]

            input (one item):
                {
                "timestamp": [100.0, 101.0, 102.0],
                "f1": [1, 2, 3],
                "f2": [b"a", b"b", b"c"],
                "i1": 10,
                "i2": b"x",
                }

        Output (two items)
            # Feature "f1"
            ((10, b"x", 0), ([100.0, 101.0, 102.0], [1, 2, 3])
            # Feature "f2"
            ((10, b"x", 1), ([100.0, 101.0, 102.0], [b"a", b"b", b"c"])
    """

    timestamps = input[timestamp_key]
    if (
        not isinstance(timestamps, np.ndarray)
        or timestamps.dtype.type != np.float64
    ):
        raise ValueError(
            f"Timestamp with value {timestamps} is expected to be np.float64"
            f" numpy array, but has dtype {type(timestamps)} instead."
        )

    index = []
    for index_schema in schema.indexes:
        expected_type = tp_dtype_to_py_type(index_schema.dtype)
        src_value = input[index_schema.name]

        if not isinstance(src_value, expected_type):
            raise ValueError(
                f'Index "{index_schema.name}" with value "{src_value}" is'
                f" expected to be of type {expected_type} (since the Temporian "
                f" dtype is {index_schema.dtype}) but type"
                f" {type(src_value)} was found."
            )
        index.append(src_value)
    index_tuple = tuple(index)

    for feature_idx, feature_schema in enumerate(schema.features):
        expected_dtype = tp_dtype_to_np_dtype(feature_schema.dtype)
        src_value = input[feature_schema.name]

        if (
            not isinstance(src_value, np.ndarray)
            or src_value.dtype.type != expected_dtype
        ):
            if isinstance(src_value, np.ndarray):
                effective_type = src_value.dtype.type
            else:
                effective_type = type(src_value)

            raise ValueError(
                f'Feature "{feature_schema.name}" with value "{src_value}" is'
                " expected to by a numpy array with dtype"
                f" {expected_dtype} (since the Temporian dtype is"
                f" {feature_schema.dtype}) but numpy dtype"
                f" {effective_type} was found."
            )

        yield index_tuple + (feature_idx,), (timestamps, src_value)

    if len(schema.features) == 0:
        yield index_tuple + (-1,), (timestamps, None)


@beam.ptransform_fn
def to_event_set(
    pipe: beam.PCollection[Dict[str, Any]],
    schema: Schema,
    timestamp_key: str = "timestamp",
    grouped_by_index: bool = True,
) -> PEventSet:
    """Converts a PCollection of key:value to a Beam EventSet.

    This method is compatible with the output of `from_csv_raw` and the
    Official Beam IO connectors.

    When importing data from csv files, use `from_csv` to convert csv files
    directly into EventSets.

    Unlike Temporian in-process EventSet import method (
    [tp.event_set][temporian.event_set])), this method (`tpb.to_event_set`)
    requires for timestamps to be numerical values.

    Args:
        pipe: Beam pipe of key values.
        schema: Schema of the data. Note: The schema of a Temporian node is
            available with `node.schema`.
        timestamp_key: Key containing the timestamps.
        grouped_by_index: Whether events are grouped by index. Run
            `tp.help.grouped_by_index()` for the documentation.

    Returns:
        PCollection of Beam EventSet.
    """

    # TODO: Add support for datetime timestamps.
    if grouped_by_index:
        return (
            pipe
            | "Parse dict"
            >> beam.FlatMap(_event_set_dict_to_event_set, schema, timestamp_key)
            | "Shuffle" >> beam.Reshuffle()
        )
    else:
        return (
            pipe
            | "Structure"
            >> beam.Map(_reindex_by_integer, schema, timestamp_key)
            # Group by index values and feature index
            | "Aggregate" >> beam.GroupByKey()
            # Build feature and timestamps arrays.
            | "Merge timestamps"
            >> beam.ParDo(_MergeTimestampsSplitFeatures(len(schema.features)))
            | "Shuffle" >> beam.Reshuffle()
        )


def _convert_to_dict_event_key_value(
    item: Tuple[
        Tuple[BeamIndex, ...],
        Iterable[IndexValue],
    ],
    schema: Schema,
    timestamp_key: str,
) -> Iterator[Dict[str, Any]]:
    index, feature_blocks = item

    # Sort the feature by feature index.
    # The feature index is the last value (-1) of the key (first element of the
    # tuple).
    feature_blocks = sorted(feature_blocks, key=lambda x: x[0][-1])
    assert len(feature_blocks) > 0

    # All the feature blocks have the same timestamps. We use the first one.
    common_item_dict = {}
    for index_schema, index_value in zip(schema.indexes, index):
        common_item_dict[index_schema.name] = index_value

    timestamps = feature_blocks[0][1][0]
    for event_idx, timestamp in enumerate(timestamps):
        item_dict = common_item_dict.copy()
        item_dict[timestamp_key] = timestamp
        for feature_schema, feature in zip(schema.features, feature_blocks):
            item_dict[feature_schema.name] = feature[1][1][event_idx]

        yield item_dict


def _convert_to_dict_event_set_key_value(
    item: Tuple[
        Tuple[BeamIndex, ...],
        Iterable[IndexValue],
    ],
    schema: Schema,
    timestamp_key: str,
) -> Dict[str, Any]:
    index, feature_blocks = item

    # Sort the feature by feature index.
    # The feature index is the last value (-1) of the key (first element of the
    # tuple).
    feature_blocks = sorted(feature_blocks, key=lambda x: x[0][-1])
    assert len(feature_blocks) > 0

    # All the feature blocks have the same timestamps. We use the first one.
    item_dict = {}
    for index_schema, index_value in zip(schema.indexes, index):
        item_dict[index_schema.name] = index_value

    item_dict[timestamp_key] = feature_blocks[0][1][0]
    for feature_schema, feature in zip(schema.features, feature_blocks):
        item_dict[feature_schema.name] = feature[1][1]

    return item_dict


@beam.ptransform_fn
def to_dict(
    pipe: PEventSet,
    schema: Schema,
    timestamp_key: str = "timestamp",
    grouped_by_index: bool = True,
) -> beam.PCollection[Dict[str, Any]]:
    """Converts a Beam EventSet to PCollection of key->value.

    This method is compatible with the output of `from_csv_raw` and the
    Official Beam IO connectors. This method is the inverse of `to_event_set`.

    Args:
        pipe: PCollection of Beam EventSet.
        schema: Schema of the data.
        timestamp_key: Key containing the timestamps.
        grouped_by_index: Whether events are grouped by index. Run
            `tp.help.grouped_by_index()` for the documentation.

    Returns:
        Beam pipe of key values.
    """

    # TODO: Add support for datetime timestamps.

    if grouped_by_index:
        return (
            pipe
            | "Group by features" >> beam.GroupBy(lambda x: x[0][0:-1])
            | "Convert to dict"
            >> beam.Map(
                _convert_to_dict_event_set_key_value, schema, timestamp_key
            )
        )
    else:
        return (
            pipe
            | "Group by features" >> beam.GroupBy(lambda x: x[0][0:-1])
            | "Convert to dict"
            >> beam.FlatMap(
                _convert_to_dict_event_key_value, schema, timestamp_key
            )
        )
