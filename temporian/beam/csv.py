"""Reads csv files in Beam."""

import apache_beam as beam
import csv
from apache_beam.io.fileio import MatchFiles
from temporian.core.data.node import Schema


def _parse_csv_file(file):
    with beam.io.filesystems.FileSystems.open(file.path) as byte_stream:
        string_stream = (x.decode("utf-8") for x in byte_stream)
        for row in csv.DictReader(string_stream):
            yield row


@beam.ptransform_fn
def read_csv(pipe, file_pattern: str):
    return (
        pipe
        | "List files" >> MatchFiles(file_pattern)
        | "Parse file" >> beam.FlatMap(_parse_csv_file)
    )

@beam.ptransform_fn
def read_csv_with_schema(pipe, file_pattern: str, schema: Schema):
    pass