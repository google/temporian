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

from temporal_feature_processor.implementation.pandas.operators import assign
from temporal_feature_processor.implementation.pandas.operators.tests.assign.test_data import different_index
from temporal_feature_processor.implementation.pandas.operators.tests.assign.test_data import repeated_timestamps
from temporal_feature_processor.implementation.pandas.operators.tests.assign.test_data import with_idx_more_timestamps
from temporal_feature_processor.implementation.pandas.operators.tests.assign.test_data import with_idx_same_timestamps


class AssignOperatorTest(absltest.TestCase):
  base_data_dir = "temporal_feature_processor/tests/operators/assign/data"

  def setUp(self) -> None:
    self.operator = assign.PandasAssignOperator()

  def test_different_index(self) -> None:
    self.assertRaisesRegex(
        IndexError,
        "Assign sequences must have the same index names.",
        self.operator,
        different_index.INPUT_1,
        different_index.INPUT_2,
    )

  def test_repeated_timestamps(self) -> None:
    self.assertRaisesRegex(
        ValueError,
        "Cannot have repeated timestamps in assigned EventSequence.",
        self.operator,
        repeated_timestamps.INPUT_1,
        repeated_timestamps.INPUT_1,
    )

  def test_with_idx_same_timestamps(self) -> None:
    operator_output = self.operator(
        with_idx_same_timestamps.INPUT_1, with_idx_same_timestamps.INPUT_2
    )
    self.assertEqual(
        True, with_idx_same_timestamps.OUTPUT.equals(operator_output["output"])
    )

  def test_with_idx_more_timestamps(self) -> None:
    operator_output = self.operator(
        with_idx_more_timestamps.INPUT_1, with_idx_more_timestamps.INPUT_2
    )
    self.assertEqual(
        True, with_idx_more_timestamps.OUTPUT.equals(operator_output["output"])
    )


if __name__ == "__main__":
  absltest.main()
