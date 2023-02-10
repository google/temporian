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

from temporian.implementation.numpy.operators import assign
from temporian.implementation.numpy.operators.tests.assign.test_data import (
    different_index,
)
from temporian.implementation.numpy.operators.tests.assign.test_data import (
    with_idx_more_timestamps,
)

from temporian.implementation.numpy.operators.tests.assign.test_data import (
    with_idx_same_timestamps,
)

from temporian.implementation.numpy.operators.tests.assign.test_data import (
    less_assigned_indexes,
)

from temporian.implementation.numpy.operators.tests.assign.test_data import (
    complete_timestamps,
)


class AssignOperatorTest(absltest.TestCase):
    def setUp(self) -> None:
        self.operator = assign.NumpyAssignOperator()

    def test_different_index(self) -> None:
        self.assertRaisesRegex(
            ValueError,
            "Assign sequences must have the same index names.",
            self.operator,
            different_index.INPUT_1,
            different_index.INPUT_2,
        )

    def test_with_idx_more_timestamps(self) -> None:
        operator_output = self.operator(
            with_idx_more_timestamps.INPUT_1,
            with_idx_more_timestamps.INPUT_2,
        )
        self.assertEqual(
            True,
            with_idx_more_timestamps.OUTPUT == operator_output["event"],
        )

    def test_with_idx_same_timestamps(self) -> None:
        operator_output = self.operator(
            with_idx_same_timestamps.INPUT_1,
            with_idx_same_timestamps.INPUT_2,
        )
        self.assertEqual(
            True,
            with_idx_same_timestamps.OUTPUT == operator_output["event"],
        )

    def test_less_assigned_indexes(self) -> None:
        operator_output = self.operator(
            less_assigned_indexes.INPUT_1,
            less_assigned_indexes.INPUT_2,
        )

        self.assertEqual(
            True,
            less_assigned_indexes.OUTPUT == operator_output["event"],
        )

    def test_complete_timestamps(self) -> None:
        operator_output = self.operator(
            complete_timestamps.INPUT_1,
            complete_timestamps.INPUT_2,
        )

        self.assertEqual(
            True,
            complete_timestamps.OUTPUT == operator_output["event"],
        )


if __name__ == "__main__":
    absltest.main()
