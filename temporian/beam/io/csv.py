"""Utilities to import/export Beam-Event-Set from/to dataset containers."""

from typing import Iterable, Dict, Tuple, List

from enum import Enum
import csv
import io
import numpy as np
import apache_beam as beam
from apache_beam.io.fileio import MatchFiles

from temporian.core.data.node import Schema
from temporian.beam.io.dict import (
    to_event_set,
    PEventSet,
    BeamIndex,
    IndexValue,
)


def _parse_csv_file(
    file: beam.io.filesystem.FileMetadata,
) -> Iterable[Dict[str, str]]:
    """Parse a csv file into dictionary of key -> value."""

    with beam.io.filesystems.FileSystems.open(file.path) as byte_stream:
        string_stream = (x.decode("utf-8") for x in byte_stream)
        for row in csv.DictReader(string_stream):
            yield row


@beam.ptransform_fn
def from_csv_raw(pipe, file_pattern: str) -> beam.PCollection[Dict[str, str]]:
    """Reads a file or set of csv files into a PCollection of key->values.

    This format is similar to output of the official beam IO connectors:
    https://beam.apache.org/documentation/io/connectors/

    CSV values are always string, so the output of `from_csv_raw` is always
    a dictionary of string to string. Use `to_event_set` (or better, use
    `from_csv` instead of `from_csv_raw`) to cast values to the expected
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


@beam.ptransform_fn
def from_csv(
    pipe, file_pattern: str, schema: Schema, timestamp_key: str = "timestamp"
) -> PEventSet:
    """Reads a file or set of csv files into a Beam EventSet.

    Limitation: Timestamps have to be numerical values. See documentation of
    `to_event_set` for more details.

    Usage example:

    ```
    input_node: tp.EventSetNode = ...
    p | tpb.from_csv("/tmp/path.csv", input_node.schema) | ...
    ```

    `from_csv` is equivalent to `from_csv_raw + to_event_set`.

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
        | "Read csv" >> from_csv_raw(file_pattern)
        | "Convert to Event Set"
        >> to_event_set(schema, timestamp_key, grouped_by_index=False)
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
def to_csv(
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
      | tpb.from_csv("/input.csv", input_node.schema)
      | ... # processing
      | tpb.to_csv("/output.csv", output_node.schema)
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
