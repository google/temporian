"""Utilities to import/export Beam-Event-Set from/to dataset containers."""

from typing import Iterable, Dict, Any, Tuple, Union, Optional

import csv
import io
import numpy as np
import apache_beam as beam
from apache_beam.io.fileio import MatchFiles
from temporian.core.data.node import Schema
from temporian.core.data.dtype import DType
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
BeamIndex = Union[int, float, str, bool]

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
    row: Dict[str, str], schema: Schema, timestamp_key: str
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


@beam.ptransform_fn
def to_event_set(
    pipe: beam.PCollection[Dict[str, Any]],
    schema: Schema,
    timestamp_key: str = "timestamp",
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

    Returns:
        PCollection of Beam EventSet.
    """

    # TODO: Add support for datetime timestamps.

    return (
        pipe
        | "Structure" >> beam.Map(_reindex_by_integer, schema, timestamp_key)
        # Group by index values and feature index
        | "Aggregate" >> beam.GroupByKey()
        # Build feature and timestamps arrays.
        | "Merge timestamps"
        >> beam.ParDo(_MergeTimestampsSplitFeatures(len(schema.features)))
    )


@beam.ptransform_fn
def read_csv(
    pipe, file_pattern: str, schema: Schema, timestamp_key: str = "timestamp"
) -> PEventSet:
    """Reads a file or set of csv files into a Beam EventSet.

    Limitation: Timestamps have to be numerical values. See documentation of
    `to_event_set` for more details.

    Usage example:

    ```
    input_node: tp.Node = ...
    p | tpb.read_csv("/tmp/path.csv", input_node.schema) | ...
    ```

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


def _convert_to_csv(
    item: Tuple[
        Tuple[BeamIndex, ...],
        Iterable[IndexValue],
    ]
) -> str:
    index, feature_blocks = item
    index_data = list(index)

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
        feature_data = [f[1][1][event_idx] for f in feature_blocks]
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
    input_node: tp.Node = ...
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
