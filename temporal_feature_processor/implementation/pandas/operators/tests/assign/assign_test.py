from absl.testing import absltest

import test_data
from temporal_feature_processor.implementation.pandas.operators import PandasAssignOperator


class AssignOperatorTest(absltest.TestCase):
  base_data_dir = "temporal_feature_processor/tests/operators/assign/data"

  def setUp(self) -> None:
    self.operator = PandasAssignOperator()

  def test_different_index(self) -> None:
    self.assertRaises(IndexError, self.operator,
                      test_data.different_index.INPUT_1,
                      test_data.different_index.INPUT_2)

  def test_repeated_timestamps(self) -> None:
    self.assertRaises(ValueError, self.operator,
                      test_data.repeated_timestamps.INPUT_1,
                      test_data.repeated_timestamps.INPUT_1)

  def test_with_idx_same_timestamps(self) -> None:
    operator_output = self.operator(test_data.with_idx_same_timestamps.INPUT_1,
                                    test_data.with_idx_same_timestamps.INPUT_2)
    self.assertEqual(
        True, test_data.with_idx_same_timestamps.OUTPUT.equals(operator_output))

  def test_with_idx_more_timestamps(self) -> None:
    operator_output = self.operator(test_data.with_idx_more_timestamps.INPUT_1,
                                    test_data.with_idx_more_timestamps.INPUT_2)
    self.assertEqual(
        True, test_data.with_idx_more_timestamps.OUTPUT.equals(operator_output))


if __name__ == "__main__":
  absltest.main()
