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

from absl.testing import absltest
from numpy.testing import assert_array_equal

from temporian.io.tensorflow import to_tensorflow
from temporian.implementation.numpy.data.io import event_set


class TensorFlowTest(absltest.TestCase):
    def test_base(self) -> None:
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
        tf_dataset = to_tensorflow(evset).batch(4)
        num_rows = 0
        for row in tf_dataset:
            for key, value in data_dict.items():
                assert_array_equal(row[key], value)
            assert_array_equal(row["timestamp"], [1, 2, 3, 4])
            num_rows += 1

        self.assertEqual(num_rows, 1)


if __name__ == "__main__":
    absltest.main()
