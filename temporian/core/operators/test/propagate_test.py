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

from temporian.core.data.node import input_node
from temporian.core.data.dtype import DType
from temporian.core.operators.propagate import propagate


class PropagateOperatorTest(absltest.TestCase):
    def setUp(self):
        pass

    def test_basic(self):
        node = input_node(
            [
                ("a", DType.FLOAT64),
                ("b", DType.FLOAT64),
            ],
            indexes=[("x", DType.STRING)],
            is_unix_timestamp=False,
        )
        sampling = input_node(
            [],
            indexes=[("x", DType.STRING), ("y", DType.STRING)],
            is_unix_timestamp=False,
        )
        _ = propagate(input=node, sampling=sampling)

    def test_error_wrong_index(self):
        node = input_node(
            [
                ("a", DType.FLOAT64),
                ("b", DType.FLOAT64),
            ],
            indexes=[("z", DType.STRING)],
            is_unix_timestamp=False,
        )
        sampling = input_node(
            [],
            indexes=[("x", DType.STRING), ("y", DType.STRING)],
            is_unix_timestamp=False,
        )
        with self.assertRaisesRegex(
            ValueError,
            (
                "The indexes of input should be contained in the indexes of"
                " sampling"
            ),
        ):
            _ = propagate(input=node, sampling=sampling)

    def test_error_wrong_index_type(self):
        node = input_node(
            [
                ("a", DType.FLOAT64),
                ("b", DType.FLOAT64),
            ],
            indexes=[("x", DType.INT32)],
            is_unix_timestamp=False,
        )
        sampling = input_node(
            [],
            indexes=[("x", DType.STRING), ("y", DType.STRING)],
            is_unix_timestamp=False,
        )
        with self.assertRaisesRegex(
            ValueError,
            "However, the dtype is different",
        ):
            _ = propagate(input=node, sampling=sampling)


if __name__ == "__main__":
    absltest.main()
