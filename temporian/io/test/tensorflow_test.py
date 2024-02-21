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

import os
import tempfile

from absl.testing import absltest
from numpy.testing import assert_array_equal
import tensorflow as tf
from temporian.test.utils import assertEqualEventSet

from temporian.io.tensorflow import (
    to_tensorflow_dataset,
    to_tensorflow_record,
    from_tensorflow_record,
)
from temporian.implementation.numpy.data.io import event_set
from temporian.core.data import schema
from temporian.core.operators.combine import combine


class TensorFlowTest(absltest.TestCase):
    def test_to_tensorflow_dataset(self) -> None:
        data_dict = {
            "f1": [10, 11, 12, 13],
            "f2": [0.1, 0.2, 0.3, 0.4],
            "f3": [b"a", b"b", b"c", b"d"],
            "i1": [1, 1, 2, 2],
            "i2": [b"x", b"x", b"x", b"y"],
        }

        evset = event_set(
            timestamps=[1, 2, 3, 4],
            features=data_dict,
            indexes=["i1", "i2"],
        )
        tf_dataset = to_tensorflow_dataset(evset).batch(4)
        num_rows = 0
        for row in tf_dataset:
            for key, value in data_dict.items():
                assert_array_equal(row[key], value)
            assert_array_equal(row["timestamp"], [1, 2, 3, 4])
            num_rows += 1

        self.assertEqual(num_rows, 1)

    def test_to_tensorflow_record_grouped_by_index(self) -> None:
        data_dict = {
            "f1": [10, 11, 12, 13],
            "f2": [0.1, 0.2, 0.3, 0.4],
            "f3": [b"a", b"b", b"c", b"d"],
            "i1": [1, 1, 2, 2],
            "i2": [b"x", b"x", b"y", b"y"],
        }

        evset = event_set(
            timestamps=[1, 2, 3, 4],
            features=data_dict,
            indexes=["i1", "i2"],
        )

        tmp_dir_handle = tempfile.TemporaryDirectory()
        tmp_file = os.path.join(tmp_dir_handle.name, "data")

        to_tensorflow_record(evset, path=tmp_file, format="grouped_by_index")

        expected_1 = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "timestamp": tf.train.Feature(
                        float_list=tf.train.FloatList(value=[3, 4])
                    ),
                    "i1": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[2])
                    ),
                    "i2": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b"y"])
                    ),
                    "f1": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[12, 13])
                    ),
                    "f2": tf.train.Feature(
                        float_list=tf.train.FloatList(value=[0.3, 0.4])
                    ),
                    "f3": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b"c", b"d"])
                    ),
                }
            )
        )

        expected_2 = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "timestamp": tf.train.Feature(
                        float_list=tf.train.FloatList(value=[1, 2])
                    ),
                    "i1": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[1])
                    ),
                    "i2": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b"x"])
                    ),
                    "f1": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[10, 11])
                    ),
                    "f2": tf.train.Feature(
                        float_list=tf.train.FloatList(value=[0.1, 0.2])
                    ),
                    "f3": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[b"a", b"b"])
                    ),
                }
            )
        )

        self.assertEqual(_extract_tfrecord(tmp_file), [expected_1, expected_2])

    def test_from_tensorflow_record(self) -> None:
        data_dict = {
            "f1": [10, 11, 12, 13],
            "f2": [0.1, 0.2, 0.3, 0.4],
            "f3": [b"a", b"b", b"c", b"d"],
            "i1": [1, 1, 2, 2],
            "i2": [b"x", b"x", b"y", b"y"],
        }

        evset = event_set(
            timestamps=[1, 2, 3, 4],
            features=data_dict,
            indexes=["i1", "i2"],
        )

        tmp_dir_handle = tempfile.TemporaryDirectory()
        tmp_file = os.path.join(tmp_dir_handle.name, "data")

        to_tensorflow_record(evset, path=tmp_file, format="grouped_by_index")
        loaded_evtset = from_tensorflow_record(
            path=tmp_file, schema=evset.schema
        )
        assertEqualEventSet(self, evset, loaded_evtset)

    def test_from_sharded_tensorflow_record(self) -> None:
        evset_1 = event_set(
            timestamps=[1, 2],
            features={"f1": [10, 11], "i1": [1, 1]},
            indexes=["i1"],
        )
        evset_2 = event_set(
            timestamps=[3, 4],
            features={"f1": [12, 13], "i1": [2, 2]},
            indexes=["i1"],
        )
        evset_all = combine(evset_1, evset_2)

        tmp_dir_handle = tempfile.TemporaryDirectory()
        tmp_file_1 = os.path.join(tmp_dir_handle.name, "data_00000_of_00001")
        tmp_file_2 = os.path.join(tmp_dir_handle.name, "data_00001_of_00001")

        to_tensorflow_record(
            evset_1, path=tmp_file_1, format="grouped_by_index"
        )
        to_tensorflow_record(
            evset_2, path=tmp_file_2, format="grouped_by_index"
        )
        loaded_evtset = from_tensorflow_record(
            path=[tmp_file_1, tmp_file_2], schema=evset_all.schema
        )
        assertEqualEventSet(self, evset_all, loaded_evtset)

    def test_from_tensorflow_record_non_sorted(self) -> None:
        tmp_dir_handle = tempfile.TemporaryDirectory()
        tmp_file = os.path.join(tmp_dir_handle.name, "data")

        with tf.io.TFRecordWriter(tmp_file, options="GZIP") as file_writer:
            ex = tf.train.Example()
            ex.features.feature["timestamp"].float_list.value[:] = [1, 3, 2]
            file_writer.write(ex.SerializeToString())

        with self.assertRaisesRegex(
            ValueError, "The timestamps are not sorted"
        ):
            _ = from_tensorflow_record(
                path=tmp_file,
                schema=schema.Schema(
                    features=[], indexes=[], is_unix_timestamp=False
                ),
            )


def _extract_tfrecord(path: str):
    result = []
    tf_dataset = tf.data.TFRecordDataset(path, compression_type="GZIP")
    for serialized_example in tf_dataset:
        example = tf.train.Example()
        example.ParseFromString(serialized_example.numpy())
        result.append(example)
    return result


if __name__ == "__main__":
    absltest.main()
