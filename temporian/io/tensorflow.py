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

from temporian.implementation.numpy.data.event_set import EventSet
from temporian.core.operators.drop_index import drop_index


def to_tensorflow(
    evset: EventSet, timestamps: str = "timestamp"
) -> "tensorflow.data.Dataset":
    """Converts an [`EventSet`][temporian.EventSet] to a tensorflow Dataset.

    Usage example:
        ```python
        evtset = event_set(
            timestamps=[1, 2, 3, 4],
            features={
                "f1": [10, 11, 12, 13],
                "f2": [b"a", b"b", b"c", b"d"],
                "label": [0, 1, 0, 1],
            },
        )

        tf_dataset = tp.to_tensorflow(evtset)

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

    import tensorflow as tf

    if len(evset.schema.indexes) != 0:
        evset = drop_index(evset)

    data = evset.get_arbitrary_index_data()

    dict_data = {timestamps: data.timestamps}

    for feature_idx, feature in enumerate(evset.schema.features):
        dict_data[feature.name] = data.features[feature_idx]

    return tf.data.Dataset.from_tensor_slices(dict_data)
