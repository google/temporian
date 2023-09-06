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
from tensorflow.core.example import example_pb2
from temporian.implementation.numpy.operators.test.test_util import (
    assertEqualEventSet,
)

from temporian.io.tensorflow import (
    to_tensorflow_dataset,
    to_tensorflow_record,
    from_tensorflow_record,
)
from temporian.implementation.numpy.data.io import event_set


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

        self.assertEqual(
            str(_extract_tfrecord(tmp_file)),
            """[features {
  feature {
    key: "timestamp"
    value {
      float_list {
        value: 3
        value: 4
      }
    }
  }
  feature {
    key: "i2"
    value {
      bytes_list {
        value: "y"
      }
    }
  }
  feature {
    key: "i1"
    value {
      int64_list {
        value: 2
      }
    }
  }
  feature {
    key: "f3"
    value {
      bytes_list {
        value: "c"
        value: "d"
      }
    }
  }
  feature {
    key: "f2"
    value {
      float_list {
        value: 0.3
        value: 0.4
      }
    }
  }
  feature {
    key: "f1"
    value {
      int64_list {
        value: 12
        value: 13
      }
    }
  }
}
, features {
  feature {
    key: "timestamp"
    value {
      float_list {
        value: 1
        value: 2
      }
    }
  }
  feature {
    key: "i2"
    value {
      bytes_list {
        value: "x"
      }
    }
  }
  feature {
    key: "i1"
    value {
      int64_list {
        value: 1
      }
    }
  }
  feature {
    key: "f3"
    value {
      bytes_list {
        value: "a"
        value: "b"
      }
    }
  }
  feature {
    key: "f2"
    value {
      float_list {
        value: 0.1
        value: 0.2
      }
    }
  }
  feature {
    key: "f1"
    value {
      int64_list {
        value: 10
        value: 11
      }
    }
  }
}
]""",
        )

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
