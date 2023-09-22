"""Utilities to import/export Beam-Event-Set from/to dataset containers."""

from typing import Dict, Any, Iterator

import numpy as np
import apache_beam as beam

from temporian.beam.io.dict import to_event_set, to_dict
from temporian.core.data.dtype import DType, tp_dtype_to_py_type
from temporian.core.data.node import Schema
from temporian.implementation.numpy.data.dtype_normalization import (
    tp_dtype_to_np_dtype,
)
from temporian.io.format import (
    TFRecordEventSetFormat,
    TFRecordEventSetFormatChoices,
)
from temporian.io.tensorflow import import_tf
from temporian.beam.typing import BeamEventSet


class _TFExampleToDict(beam.DoFn):
    def __init__(self, schema: Schema, timestamp_key: str):
        self._schema = schema
        self._timestamp_key = timestamp_key

    def process(self, example: "tf.train.Example") -> Iterator[Dict[str, Any]]:
        dict_example = {}

        def get_value(key):
            if key not in example.features.feature:
                raise ValueError(f"Missing feature {key}")

            feature = example.features.feature[key]
            if feature.HasField("int64_list"):
                return feature.int64_list.value
            elif feature.HasField("float_list"):
                return feature.float_list.value
            elif feature.HasField("bytes_list"):
                return feature.bytes_list.value
            else:
                raise ValueError("Non supported type")

        timestamps = np.array(get_value(self._timestamp_key), np.float64)
        dict_example[self._timestamp_key] = timestamps
        num_events = len(timestamps)

        for feature in self._schema.features:
            value = get_value(feature.name)
            if len(value) != num_events:
                raise ValueError(
                    f"Timestamp '{self._timestamp_key}' and feature"
                    f" '{feature.name}' should contain the same number of"
                    f" values. Timestamp '{self._timestamp_key}' contains"
                    f" {num_events} values and feature '{feature.name}'"
                    f" contains {len(value)} values."
                )
            np_type = tp_dtype_to_np_dtype(feature.dtype)
            dict_example[feature.name] = np.array(value, dtype=np_type)

        for index in self._schema.indexes:
            value = get_value(index.name)
            if len(value) != 1:
                raise ValueError(
                    "Index value is expected to have exactly one value."
                    f" Instead got {value}"
                )
            py_type = tp_dtype_to_py_type(index.dtype)
            dict_example[index.name] = py_type(value[0])

        yield dict_example


class _DictToTFExample(beam.DoFn):
    def __init__(self, schema: Schema, timestamp_key: str):
        self._schema = schema
        self._timestamp_key = timestamp_key
        self._tf = import_tf()

    def process(
        self, dict_example: Dict[str, Any]
    ) -> Iterator["tensorflow.train.Example"]:
        ex = self._tf.train.Example()

        def f(example: "tensorflow.train.Example", key: str):
            return example.features.feature[key]

        # Timestamps
        f(ex, self._timestamp_key).float_list.value[:] = dict_example[
            self._timestamp_key
        ]

        # Features
        for feature_schema in self._schema.features:
            src_value = dict_example[feature_schema.name]
            if feature_schema.dtype in [
                DType.BOOLEAN,
                DType.INT32,
                DType.INT64,
            ]:
                f(ex, feature_schema.name).int64_list.value[
                    :
                ] = src_value = dict_example[feature_schema.name]

            elif feature_schema.dtype in [
                DType.FLOAT32,
                DType.FLOAT64,
            ]:
                f(ex, feature_schema.name).float_list.value[
                    :
                ] = src_value = dict_example[feature_schema.name]

            elif feature_schema.dtype == DType.STRING:
                f(ex, feature_schema.name).bytes_list.value[
                    :
                ] = src_value = dict_example[feature_schema.name]

            else:
                raise ValueError("Non supported feature dtype")

        # Indexes
        for index_schema in self._schema.indexes:
            src_value = dict_example[index_schema.name]
            if index_schema.dtype in [
                DType.BOOLEAN,
                DType.INT32,
                DType.INT64,
            ]:
                f(ex, index_schema.name).int64_list.value.append(src_value)
            elif index_schema.dtype in [
                DType.FLOAT32,
                DType.FLOAT64,
            ]:
                f(ex, index_schema.name).float_list.value.append(src_value)
            elif index_schema.dtype == DType.STRING:
                f(ex, index_schema.name).bytes_list.value.append(src_value)
            else:
                raise ValueError("Non supported index dtype")

        yield ex


@beam.ptransform_fn
def from_tensorflow_record(
    pipe,
    file_pattern: str,
    schema: Schema,
    timestamp_key: str = "timestamp",
    format: TFRecordEventSetFormatChoices = TFRecordEventSetFormat.GROUPED_BY_INDEX,
) -> BeamEventSet:
    """Imports an EventSet from a TF.Records of TF.Examples.

    TF.Records of TF.Examples is one of the standard solution to store data
    for TensorFlow models.
    https://www.tensorflow.org/tutorials/load_data/tfrecord

    The GZIP compression is used.

    Usage example:

    ```
    input_node: tp.EventSetNode = ...
    ( p
      | tpb.from_tensorflow_record("/input.tfr.gzip", input_node.schema)
      | ... # processing
      | tpb.to_tensorflow_record("/output.tfr.gzip", output_node.schema)
    )
    ```

    Args:
        pipe: Beam pipe.
        file_pattern: Path or path matching expression compatible with
            `MatchFiles`.
        schema: Schema of the data. If you have a Temporian node, the schema is
            available with `node.schema`.
        timestamp_key: Key containing the timestamps.
        format: Format of the events inside the received record. At the moment
            only TFRecordEventSetFormat.GROUPED_BY_INDEX is supported. See
            [TFRecordEventSetFormat][temporian.io.format.TFRecordEventSetFormat]
            for more.

    Returns:
        A PCollection of event-set compatible with tpb.run.
    """

    if format == TFRecordEventSetFormat.SINGLE_EVENTS:
        raise ValueError(
            "format=TFRecordEventSetFormat.SINGLE_EVENTS is not implemented"
        )
    if format != TFRecordEventSetFormat.GROUPED_BY_INDEX:
        raise ValueError(f"Unknown format {format}")

    tf = import_tf()

    return (
        pipe
        | "Read tf.record"
        >> beam.io.tfrecordio.ReadFromTFRecord(
            file_pattern=file_pattern,
            coder=beam.coders.ProtoCoder(tf.train.Example),
            compression_type=beam.io.filesystem.CompressionTypes.GZIP,
        )
        | "Tf.record to dict"
        >> beam.ParDo(
            _TFExampleToDict(
                schema=schema,
                timestamp_key=timestamp_key,
            )
        )
        | "Dict to event set"
        >> to_event_set(
            schema=schema,
            timestamp_key=timestamp_key,
            format=format,
        )
    )


@beam.ptransform_fn
def to_tensorflow_record(
    pipe: BeamEventSet,
    file_path_prefix: str,
    schema: Schema,
    timestamp_key: str = "timestamp",
    format: TFRecordEventSetFormatChoices = TFRecordEventSetFormat.GROUPED_BY_INDEX,
    **kwargs,
):
    """Export an EventSet to a TF.Records of TF.Examples.

    TF.Records of TF.Examples is one of the standard solution to store data
    for TensorFlow models.
    https://www.tensorflow.org/tutorials/load_data/tfrecord

    The GZIP compression is used.

    Usage example:

    ```
    input_node: tp.EventSetNode = ...
    ( p
      | tpb.from_tensorflow_record("/input.tfr.gzip", input_node.schema)
      | ... # processing
      | tpb.to_tensorflow_record("/output.tfr.gzip", output_node.schema)
    )
    ```

    Args:
        pipe: Beam pipe.
        file_pattern: Path or path matching expression compatible with
            `MatchFiles`.
        schema: Schema of the data. If you have a Temporian node, the schema is
            available with `node.schema`.
        timestamp_key: Key containing the timestamps.
        format: Format of the events inside the received record. At the moment
            only TFRecordEventSetFormat.GROUPED_BY_INDEX is supported. See
            [TFRecordEventSetFormat][temporian.io.format.TFRecordEventSetFormat]
            for more.
    """

    if format == TFRecordEventSetFormat.SINGLE_EVENTS:
        raise ValueError(
            "format=TFRecordEventSetFormat.SINGLE_EVENTS is not implemented"
        )
    if format != TFRecordEventSetFormat.GROUPED_BY_INDEX:
        raise ValueError(f"Unknown format {format}")

    tf = import_tf()

    return (
        pipe
        | "Event set to dict"
        >> to_dict(schema=schema, timestamp_key=timestamp_key, format=format)
        | "Dict to Tf.record"
        >> beam.ParDo(
            _DictToTFExample(schema=schema, timestamp_key=timestamp_key)
        )
        | "Write tf.record"
        >> beam.io.tfrecordio.WriteToTFRecord(
            file_path_prefix=file_path_prefix,
            coder=beam.coders.ProtoCoder(tf.train.Example),
            compression_type=beam.io.filesystem.CompressionTypes.GZIP,
            **kwargs,
        )
    )
