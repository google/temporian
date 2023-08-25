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
from temporian.beam.io.dict import PEventSet, BeamIndex


@beam.ptransform_fn
def to_tensorflow_record(
    pipe: PEventSet,
    file_path_prefix: str,
    schema: Schema,
    timestamp_key: str = "timestamp",
    grouped_by_index: bool = True,
    **wargs,
):
    pass


@beam.ptransform_fn
def from_tensorflow_record(
    pipe,
    file_pattern: str,
    schema: Schema,
    timestamp_key: str = "timestamp",
    grouped_by_index: bool = True,
) -> PEventSet:
    pass
