"""Utilities to import/export Beam-Event-Set from/to dataset containers."""

from typing import Iterable, Dict, Any, Tuple, Union, Optional, List, Iterator

from enum import Enum
import csv
import io
import numpy as np
import apache_beam as beam
from apache_beam.io.fileio import MatchFiles
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


class UserEventSetFormat(Enum):
    """Various representations of EventSets for user generation or consumption.

    When combining Temporian program with your own beam stages, select the
    most suited user event-set format compatible with your code.
    """

    eventKeyValue = "eventKeyValue"
    """
    Each event is represented as a dictionary of key to value for the features,
    the indexes and the timestamp. Values are python primitives matching the
    schema e.g. a `tp.int64` is feed as an `int`. Non-matching primitives are
    casted e.g. a int is casted into a float.

    For example:
        Schema
            features=[("f1", DType.INT64), ("f2", DType.STRING)]
            indexes=[("i1", DType.INT64), ("i2", DType.STRING)]

        Data:
            {"timestamp": 100.0, "f1": 1, "f2": b"a", "i1": 10, "i2": b"x"}
    """

    eventSetKeyValue = "eventSetKeyValue"
    """
    All the events in the same index are represented together in a dictionary of
    key to values for the features, the indexes and the timestamp. Timestamps
    are sorted. Values, index and timestamps are stored in Numpy arrays.

    Index values are python primitives matching the schema e.g. a `tp.int64` is
    feed as an `int`. Timestamps are a numpy array of float64. Feature values
    are numpy array matching the schema e.g. a `tp.int64` is feed as a numpy
    array of np.int64.

    For example:

        Schema
            features=[("f1", DType.INT64), ("f2", DType.STRING)]
            indexes=[("i1", DType.INT64), ("i2", DType.STRING)]

        Data:
            {
            "timestamp": np.array([100.0, 101.0, 102.0]),
            "f1": np.array([1, 2, 3]),
            "f2": np.array([b"a", b"b", b"c"]),
            "i1": 10,
            "i2": b"x",
            }
    """


def _parse_csv_file(
    file: beam.io.filesystem.FileMetadata,
) -> Iterable[Dict[str, str]]:
    """Parse a csv file into dictionary of key -> value."""

    with beam.io.filesystems.FileSystems.open(file.path) as byte_stream:
        string_stream = (x.decode("utf-8") for x in byte_stream)
        for row in csv.DictReader(string_stream):
            yield row


@beam.ptransform_fn
def read_csv_raw(pipe, file_pattern: str) -> beam.PCollection[Dict[str, str]]:
    """Reads a file or set of csv files into a PCollection of key->values.

    This format is similar to output of the official beam IO connectors:
    https://beam.apache.org/documentation/io/connectors/

    CSV values are always string, so the output of `read_csv_raw` is always
    a dictionary of string to string. Use `to_event_set` (or better, use
    `read_csv` instead of `read_csv_raw`) to cast values to the expected
    pipeline input dtype.

    Args:
        pipe: A begin Beam pipe.
        file_pattern: Path or path matching expression compatible with
            `MatchFiles`.

    Returns:
        A PCollection of dictionary of key:value.
    """

    return (
        pipe
        | "List files" >> MatchFiles(file_pattern)
        | "Shuffle" >> beam.Reshuffle()
        | "Parse file" >> beam.FlatMap(_parse_csv_file)
    )


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
    format: UserEventSetFormat = UserEventSetFormat.eventKeyValue,
) -> PEventSet:
    """Converts a PCollection of key:value to a Beam EventSet.

    This method is compatible with the output of `read_csv_raw` and the
    Official Beam IO connectors.

    When importing data from csv files, use `read_csv` to convert csv files
    directly into EventSets.

    Unlike Temporian in-process EventSet import method (
    [tp.event_set][temporian.event_set])), this method (`tpb.to_event_set`)
    requires for timestamps to be numerical values.

    Args:
        pipe: Beam pipe of key values.
        schema: Schema of the data. Note: The schema of a Temporian node is
            available with `node.schema`.
        timestamp_key: Key containing the timestamps.
        format: Format of the input data to be converted into an event-set.

    Returns:
        PCollection of Beam EventSet.
    """

    # TODO: Add support for datetime timestamps.

    if format == UserEventSetFormat.eventKeyValue:
        return (
            pipe
            | "Structure"
            >> beam.Map(_reindex_by_integer, schema, timestamp_key)
            # Group by index values and feature index
            | "Aggregate" >> beam.GroupByKey()
            # Build feature and timestamps arrays.
            | "Merge timestamps"
            >> beam.ParDo(_MergeTimestampsSplitFeatures(len(schema.features)))
        )
    elif format == UserEventSetFormat.eventSetKeyValue:
        return pipe | "Parse dict" >> beam.FlatMap(
            _event_set_dict_to_event_set, schema, timestamp_key
        )
    else:
        raise ValueError(f"Unknown format {format}")


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
    format: UserEventSetFormat = UserEventSetFormat.eventKeyValue,
) -> beam.PCollection[Dict[str, Any]]:
    """Converts a Beam EventSet to PCollection of key->value.

    This method is compatible with the output of `read_csv_raw` and the
    Official Beam IO connectors. This method is the inverse of `to_event_set`.

    Args:
        pipe: PCollection of Beam EventSet.
        schema: Schema of the data.
        timestamp_key: Key containing the timestamps.
        format: Format of the output data.

    Returns:
        Beam pipe of key values.
    """

    # TODO: Add support for datetime timestamps.

    if format == UserEventSetFormat.eventKeyValue:
        return (
            pipe
            | "Group by features" >> beam.GroupBy(lambda x: x[0][0:-1])
            | "Convert to dict"
            >> beam.FlatMap(
                _convert_to_dict_event_key_value, schema, timestamp_key
            )
        )
    elif format == UserEventSetFormat.eventSetKeyValue:
        return (
            pipe
            | "Group by features" >> beam.GroupBy(lambda x: x[0][0:-1])
            | "Convert to dict"
            >> beam.Map(
                _convert_to_dict_event_set_key_value, schema, timestamp_key
            )
        )
    else:
        raise ValueError(f"Unknown format {format}")


@beam.ptransform_fn
def read_csv(
    pipe, file_pattern: str, schema: Schema, timestamp_key: str = "timestamp"
) -> PEventSet:
    """Reads a file or set of csv files into a Beam EventSet.

    Limitation: Timestamps have to be numerical values. See documentation of
    `to_event_set` for more details.

    Usage example:

    ```
    input_node: tp.EventSetNode = ...
    p | tpb.read_csv("/tmp/path.csv", input_node.schema) | ...
    ```

    `read_csv` is equivalent to `read_csv_raw + to_event_set`.

    Args:
        pipe: Begin Beam pipe.
        file_pattern: Path or path matching expression compatible with
            `MatchFiles`.
        schema: Schema of the data. If you have a Temporian node, the schema is
            available with `node.schema`.
        timestamp_key: Key containing the timestamps.

    Returns:
        A PCollection of dictionary of key:value.
    """
    return (
        pipe
        | "Read csv" >> read_csv_raw(file_pattern)
        | "Convert to Event Set" >> to_event_set(schema, timestamp_key)
    )


def _bytes_to_strs(list: List) -> List:
    return [x.decode() if isinstance(x, bytes) else x for x in list]


def _convert_to_csv(
    item: Tuple[
        Tuple[BeamIndex, ...],
        Iterable[IndexValue],
    ]
) -> str:
    index, feature_blocks = item
    index_data = _bytes_to_strs(list(index))

    # Sort the feature by feature index.
    # The feature index is the last value (-1) of the key (first element of the
    # tuple).
    feature_blocks = sorted(feature_blocks, key=lambda x: x[0][-1])
    assert len(feature_blocks) > 0

    # All the feature blocks have the same timestamps. We use the first one.
    timestamps = feature_blocks[0][1][0]

    output = io.StringIO()
    writer = csv.writer(output)
    for event_idx, timestamp in enumerate(timestamps):
        feature_data = _bytes_to_strs(
            [f[1][1][event_idx] for f in feature_blocks]
        )
        writer.writerow([timestamp] + index_data + feature_data)

    return output.getvalue()


@beam.ptransform_fn
def write_csv(
    pipe: PEventSet,
    file_path_prefix: str,
    schema: Schema,
    timestamp_key: str = "timestamp",
    **wargs,
):
    """Writes a Beam EventSet to a file or set of csv files.

    Limitation: Timestamps are always stored as numerical values.
    TODO: Support datetime timestamps.

    Usage example:

    ```
    input_node: tp.EventSetNode = ...
    ( p
      | tpb.read_csv("/input.csv", input_node.schema)
      | ... # processing
      | tpb.write_csv("/output.csv", output_node.schema)
    )
    ```

    Args:
        pipe: Beam pipe containing an EventSet.
        file_path_prefix: Path or path matching expression compatible with
            WriteToText.
        schema: Schema of the data. If you have a Temporian node, the schema is
            available with `node.schema`.
        timestamp_key: Key containing the timestamps.
        **wargs: Arguments passed to `beam.io.textio.WriteToText`.
    """

    header_values = (
        [timestamp_key] + schema.index_names() + schema.feature_names()
    )
    header_string = io.StringIO()
    header_writer = csv.writer(header_string)
    header_writer.writerow(header_values)

    return (
        pipe
        | "Group by features" >> beam.GroupBy(lambda x: x[0][0:-1])
        | "Convert to csv" >> beam.Map(_convert_to_csv)
        | "Write csv"
        >> beam.io.textio.WriteToText(
            file_path_prefix=file_path_prefix,
            header=header_string.getvalue(),
            append_trailing_newlines=False,
            **wargs,
        )
    )
