# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for converting EventSets to TensorFlow dataset."""

from typing import List, Union
from copy import deepcopy
import logging

import numpy as np

from temporian.core.data.dtype import DType, tp_dtype_to_py_type
from temporian.core.operators.drop_index import drop_index
from temporian.implementation.numpy.data.dtype_normalization import (
    tp_dtype_to_np_dtype,
)
from temporian.implementation.numpy.data.event_set import (
    EventSet,
    Schema,
    IndexData,
)
from temporian.io.format import (
    TFRecordEventSetFormat,
    TFRecordEventSetFormatChoices,
)


def import_tf():
    try:
        import tensorflow as tf

        return tf
    except ImportError:
        logging.warning(
            "`tp.to_tensorflow_dataset()` requires for TensorFlow to be"
            " installed. Install TensorFlow with pip using `pip install"
            " temporian[tensorflow]` or `pip install tensorflow`."
        )
        raise


def to_tensorflow_dataset(
    evset: EventSet, timestamps: str = "timestamp"
) -> "tensorflow.data.Dataset":
    """Converts an [`EventSet`][temporian.EventSet] to a tensorflow Dataset.

    Usage example:
        ```python
        evset = event_set(
            timestamps=[1, 2, 3, 4],
            features={
                "f1": [10, 11, 12, 13],
                "f2": [b"a", b"b", b"c", b"d"],
                "label": [0, 1, 0, 1],
            },
        )

        tf_dataset = tp.to_tensorflow_dataset(evset)

        def extract_label(example):
            label = example.pop("label")
            return example, label
        tf_dataset = tf_dataset.map(extract_label).batch(100)

        model = ... # A Keras model
        model.fit(tf_dataset)
        ```

    Args:
        evset: Input event set.
        timestamps: Output key containing the timestamps.

    Returns:
        TensorFlow dataset created from EventSet.
    """

    tf = import_tf()

    if len(evset.schema.indexes) != 0:
        evset = drop_index(evset)

    data = evset.get_arbitrary_index_data()

    dict_data = {timestamps: data.timestamps}

    for feature_idx, feature in enumerate(evset.schema.features):
        dict_data[feature.name] = data.features[feature_idx]

    return tf.data.Dataset.from_tensor_slices(dict_data)


def to_tensorflow_record(
    evset: EventSet,
    path: str,
    timestamps: str = "timestamp",
    format: TFRecordEventSetFormatChoices = TFRecordEventSetFormat.GROUPED_BY_INDEX,
):
    """Exports an EventSet into TF.Records of TF.Examples.

    TF.Records of TF.Examples is one of the standard solution to store data
    for TensorFlow models.
    https://www.tensorflow.org/tutorials/load_data/tfrecord

    The GZIP compression is used.

    Args:
        evset: Event set to export.
        path: Path to output TF.Record.
        timestamps: Name of the output column containing timestamps.
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

    with tf.io.TFRecordWriter(path, options="GZIP") as file_writer:

        def f(example: tf.train.Example, key: str):
            return example.features.feature[key]

        for index_key, index_value in evset.data.items():
            ex = tf.train.Example()

            # Timestamps
            f(ex, timestamps).float_list.value[:] = index_value.timestamps

            # Features
            for feature_idx, feature_schema in enumerate(evset.schema.features):
                if feature_schema.dtype in [
                    DType.BOOLEAN,
                    DType.INT32,
                    DType.INT64,
                ]:
                    f(ex, feature_schema.name).int64_list.value[
                        :
                    ] = index_value.features[feature_idx]
                elif feature_schema.dtype in [
                    DType.FLOAT32,
                    DType.FLOAT64,
                ]:
                    f(ex, feature_schema.name).float_list.value[
                        :
                    ] = index_value.features[feature_idx]
                elif feature_schema.dtype == DType.STRING:
                    f(ex, feature_schema.name).bytes_list.value[
                        :
                    ] = index_value.features[feature_idx]
                else:
                    raise ValueError("Non supported feature dtype")

            # Indexes
            for index_value, index_schema in zip(
                index_key, evset.schema.indexes
            ):
                if index_schema.dtype in [
                    DType.BOOLEAN,
                    DType.INT32,
                    DType.INT64,
                ]:
                    f(ex, index_schema.name).int64_list.value.append(
                        index_value
                    )
                elif index_schema.dtype in [
                    DType.FLOAT32,
                    DType.FLOAT64,
                ]:
                    f(ex, index_schema.name).float_list.value.append(
                        index_value
                    )
                elif index_schema.dtype == DType.STRING:
                    f(ex, index_schema.name).bytes_list.value.append(
                        index_value
                    )
                else:
                    raise ValueError("Non supported index dtype")

            file_writer.write(ex.SerializeToString())


def from_tensorflow_record(
    path: Union[str, List[str]],
    schema: Schema,
    timestamps: str = "timestamp",
    format: TFRecordEventSetFormatChoices = TFRecordEventSetFormat.GROUPED_BY_INDEX,
) -> EventSet:
    """Imports an EventSet from a TF.Records of TF.Examples.

    TF.Records of TF.Examples is one of the standard solution to store data
    for TensorFlow models.
    https://www.tensorflow.org/tutorials/load_data/tfrecord

    The GZIP compression is used.

    Args:
        path: Path to TF.Record file or list of path to TF.Record files.
        timestamps: Name of the output column containing timestamps.
        format: Format of the events inside the received record. At the moment
            only TFRecordEventSetFormat.GROUPED_BY_INDEX is supported. See
            [TFRecordEventSetFormat][temporian.io.format.TFRecordEventSetFormat]
            for more.

    Returns:
        Imported EventSet.
    """

    # TODO(gbm): Automatic schema

    if format == TFRecordEventSetFormat.SINGLE_EVENTS:
        raise ValueError(
            "format=TFRecordEventSetFormat.SINGLE_EVENTS is not implemented"
        )
    if format != TFRecordEventSetFormat.GROUPED_BY_INDEX:
        raise ValueError(f"Unknown format {format}")

    tf = import_tf()
    evtset = EventSet(data={}, schema=deepcopy(schema))

    def get_value(example: tf.train.Example, key: str):
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

    tf_dataset = tf.data.TFRecordDataset(path, compression_type="GZIP")
    for serialized_example in tf_dataset:
        example = tf.train.Example()
        example.ParseFromString(serialized_example.numpy())

        # Timestamps
        timestamp_values = np.array(get_value(example, timestamps), np.float64)
        num_timestamps = len(timestamp_values)

        if not np.all(np.diff(timestamp_values) >= 0):
            print("timestamp_values:", timestamp_values)
            raise ValueError("The timestamps are not sorted")

        # Indexes
        indexes = []
        for index_schema in schema.indexes:
            value = get_value(example, index_schema.name)
            if len(value) != 1:
                raise ValueError(
                    "Index value is expected to have exactly one value."
                    f" Instead got {value}"
                )
            py_type = tp_dtype_to_py_type(index_schema.dtype)
            indexes.append(py_type(value[0]))

        # Features
        features = []
        for feature_schema in schema.features:
            value = get_value(example, feature_schema.name)
            if len(value) != num_timestamps:
                raise ValueError(
                    f"Timestamp '{timestamp_values}' and feature"
                    f" '{feature_schema.name}' should contain the same number"
                    f" of values. Timestamp '{timestamp_values}' contains"
                    f" {num_timestamps} values and feature"
                    f" '{feature_schema.name}' contains {len(value)} values."
                )
            np_type = tp_dtype_to_np_dtype(feature_schema.dtype)
            features.append(np.array(value, dtype=np_type))

        evtset.data[tuple(indexes)] = IndexData(
            timestamps=timestamp_values, features=features
        )

    return evtset
