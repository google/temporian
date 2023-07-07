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
from temporian.core.operators.join import join


class JoinOperatorTest(absltest.TestCase):
    def setUp(self):
        pass

    def test_left(self):
        input_1 = input_node([("a", DType.FLOAT64)])
        input_2 = input_node([("b", DType.FLOAT64)])
        _ = join(input_1, input_2)

    def test_left_on(self):
        input_1 = input_node([("a", DType.FLOAT64), ("c", DType.INT64)])
        input_2 = input_node([("b", DType.FLOAT64), ("c", DType.INT64)])
        _ = join(input_1, input_2, on="c")

    def test_duplicated_feature(self):
        input_1 = input_node([("a", DType.FLOAT64)])
        input_2 = input_node([("a", DType.FLOAT64)])
        with self.assertRaisesRegex(ValueError, "is defined in both inputs"):
            _ = join(input_1, input_2)

    def test_wrong_index(self):
        input_1 = input_node([("a", DType.FLOAT64)])
        input_2 = input_node(
            [("b", DType.FLOAT64)], indexes=[("x", DType.STRING)]
        )
        with self.assertRaisesRegex(
            ValueError, "Arguments don't have the same index"
        ):
            _ = join(input_1, input_2)

    def test_wrong_join(self):
        input_1 = input_node([("a", DType.FLOAT64)])
        input_2 = input_node([("b", DType.FLOAT64)])
        with self.assertRaisesRegex(ValueError, "Non supported join type"):
            _ = join(input_1, input_2, how="non existing join")

    def test_missing_on(self):
        input_1 = input_node([("a", DType.FLOAT64)])
        input_2 = input_node([("b", DType.FLOAT64)])
        with self.assertRaisesRegex(ValueError, "does not exist in left"):
            _ = join(input_1, input_2, on="c")

    def test_wrong_on_type(self):
        input_1 = input_node([("a", DType.FLOAT64), ("c", DType.FLOAT64)])
        input_2 = input_node([("b", DType.FLOAT64)])
        with self.assertRaisesRegex(ValueError, "Got float64 instead for left"):
            _ = join(input_1, input_2, on="c")


if __name__ == "__main__":
    absltest.main()
