"""Utilities to import/export Beam-Event-Set from/to dataset containers."""

from typing import Iterable, Dict, Any, Tuple, Union, Optional

import csv
import io
import numpy as np
import apache_beam as beam
from apache_beam.io.fileio import MatchFiles
from temporian.core.data.node import Schema
from temporian.core.data.dtypes.dtype import DType
from temporian.implementation.numpy.data.event_set import tp_dtype_to_np_dtype

# Type of one of the element of the index.
#
# In the numpy backend, index are represented as numpy primitives. However,
# Beam does not support numpy primitive as index. Therefore, all index are
# converted to python primitive of type "BeamIndex".
BeamIndex = Union[int, float, str, bool]

# Type of one of the element of the index with features.
#
# A BeamIndex concatenated with a feature index (int).
# The feature index -1 is used to represent the timestamps of event-set without
# features.
BeamIndexAndFeature = BeamIndex

# Structured representation of a single event / row during the conversion to
# internal Temporian beam format. In a StructuredRow, index and features are
# indexed by integer instead of string keys.
#
# Contains the index, the timestamps, and the features.
# The indexes and features are ordered according to a Schema.
StructuredRow = Tuple[
    Tuple[BeamIndex, ...], Tuple[np.float64, Tuple[np.generic, ...]]
]

# Unit of data for all operator computation.
# Contains the index+feature_idx, timestamps, and feature values.
# The feature value can be None for event-set without features.
BeamEventSet = Tuple[
    Tuple[BeamIndexAndFeature, ...], Tuple[np.ndarray, Optional[np.ndarray]]
]


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
    """Reads a csv file or set of files into a PCollection of key->values.

    This is the same format as the official beam IO connectors:
    https://beam.apache.org/documentation/io/connectors/

    Note: The values are always strings. Use `convert_to_tp_event_set` (or
    better, use `read_csv` instead of `read_csv_raw`) to convert values to the
    expected pipeline dtype.

    Args:
        pipe: Beam pipe.
        file_pattern: Path or path matching expression compatible with
            MatchFiles.

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


def _structure_fm(
    row: Dict[str, str], schema: Schema, timestamp_key: str
) -> StructuredRow:
    """Transforms a dict of key:value to a StructuredRow."""

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


class _MergeTimestampsDP(beam.DoFn):
    """Aggregates StructuredRows into BeamEventSet."""

    def __init__(self, num_features: int):
        self._num_features = num_features

    def process(
        self,
        item: Tuple[
            Tuple[BeamIndex, ...],
            Iterable[Tuple[np.float64, Tuple[np.generic, ...]]],
        ],
    ) -> Iterable[BeamEventSet]:
        index, feat_and_ts = item
        timestamps = np.array([v[0] for v in feat_and_ts], dtype=np.float64)
        for feature_idx in range(self._num_features):
            values = np.array([v[1][feature_idx] for v in feat_and_ts])
            yield index + (feature_idx,), (timestamps, values)


@beam.ptransform_fn
def convert_to_tp_event_set(
    pipe: beam.PCollection[Dict[str, str]],
    schema: Schema,
    timestamp_key: str = "timestamp",
) -> beam.PCollection[BeamEventSet]:
    """Converts a PCollection of key:value to a Beam event-set.

    Use this method when importing data using the beam i/o connectors.

    If importing data from csv files, you can use `read_csv` instead.

    Note: Timestamps have to be numerical values.
    TODO: Support datetime timestamps.

    Args:
        pipe: Beam pipe.
        schema: Schema of the data. If you have a Temporian node, the schema is
            available with `node.schema`.
        timestamp_key: key containing the timestamps.

    Returns:
        PCollection of Beam event set.
    """

    return (
        pipe
        | "Structure" >> beam.Map(_structure_fm, schema, timestamp_key)
        # Group by index values and feature index
        | "Aggregate" >> beam.GroupByKey()
        # Build feature and timestamps arrays.
        | "Merge timestamps"
        >> beam.ParDo(_MergeTimestampsDP(len(schema.features)))
    )


@beam.ptransform_fn
def read_csv(
    pipe, file_pattern: str, schema: Schema, timestamp_key: str = "timestamp"
) -> beam.PCollection[BeamEventSet]:
    """Reads a csv file or set of files into a Beam event set.

    Note: Timestamps have to be numerical values.
    TODO: Support datetime timestamps.

    Usage example:

    ```
    input_node: tp.Node = ...
    p | tp_beam.read_csv("/tmp/path.csv", input_node.schema) | ...

    ```

    Args:
        pipe: Beam pipe.
        file_pattern: Path or path matching expression compatible with
            MatchFiles.
        schema: Schema of the data. If you have a Temporian node, the schema is
            available with `node.schema`.
        timestamp_key: key containing the timestamps.

    Returns:
        A PCollection of dictionary of key:value.
    """
    return (
        pipe
        | "Read csv" >> read_csv_raw(file_pattern)
        | "Convert to Event Set"
        >> convert_to_tp_event_set(schema, timestamp_key)
    )


def _convert_to_csv(
    item: Tuple[
        Tuple[BeamIndex, ...],
        Iterable[BeamEventSet],
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
    pipe: beam.PCollection[BeamEventSet],
    file_path_prefix: str,
    schema: Schema,
    timestamp_key: str = "timestamp",
    **wargs,
):
    """Writes a Beam event set to a csv file or set of files.

    Note: Timestamps are exported as numerical values.
    TODO: Support datetime timestamps.

    Usage example:

    ```
    input_node: tp.Node = ...
    ( p
      | tp_beam.read_csv("/input.csv", input_node.schema)
      | ... # processing
      | tp_beam.write_csv("/output.csv", output_node.schema)
    )
    ```

    Args:
        pipe: Beam pipe.
        file_path_prefix: Path or path matching expression compatible with
            WriteToText.
        schema: Schema of the data. If you have a Temporian node, the schema is
            available with `node.schema`.
        timestamp_key: key containing the timestamps.
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
