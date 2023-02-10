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

from temporian.implementation.numpy.operators import sum
from temporian.implementation.numpy.operators.tests.sum.test_data import (
    correct_sum,
)
from temporian.implementation.numpy.operators.tests.sum.test_data import (
    different_timestamps,
)
from temporian.implementation.numpy.operators.tests.sum.test_data import (
    sum_with_nan,
)


class AssignOperatorTest(absltest.TestCase):
    def setUp(self) -> None:
        self.operator = sum.NumpySumOperator()

    def test_correct_sum(self) -> None:
        operator_output = self.operator(
            correct_sum.INPUT_1,
            correct_sum.INPUT_2,
        )
        self.assertEqual(
            True,
            correct_sum.OUTPUT == operator_output["event"],
        )

    def test_different_timestamps(self) -> None:
        self.assertRaisesRegex(
            ValueError,
            "Sampling of both events must be equal.",
            self.operator,
            different_timestamps.INPUT_1,
            different_timestamps.INPUT_2,
        )

    def test_different_indexes(self) -> None:
        self.assertRaisesRegex(
            ValueError,
            "Sampling of both events must be equal.",
            self.operator,
            different_timestamps.INPUT_1,
            different_timestamps.INPUT_2,
        )

    def test_sum_with_nan(self) -> None:
        operator_output = self.operator(
            sum_with_nan.INPUT_1,
            sum_with_nan.INPUT_2,
        )
        self.assertEqual(
            True,
            sum_with_nan.OUTPUT == operator_output["event"],
        )


if __name__ == "__main__":
    absltest.main()
