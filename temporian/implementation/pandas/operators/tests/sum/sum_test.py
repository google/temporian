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

from temporian.implementation.pandas.operators import sum
from temporian.implementation.pandas.operators.tests.sum.test_data import (
    different_shape,
)


class SumOperatorTest(absltest.TestCase):
    def setUp(self) -> None:
        self.operator = sum.PandasSumOperator()

    def test_different_shape(self) -> None:
        self.assertRaisesRegex(
            ValueError,
            "data_1 and data_2 must have same shape.",
            self.operator,
            different_shape.INPUT_1,
            different_shape.INPUT_2,
        )


if __name__ == "__main__":
    absltest.main()
