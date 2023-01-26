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

import pandas as pd
from absl.testing import absltest

from temporal_feature_processor.implementation.pandas.operators.window import simple_moving_average
from temporal_feature_processor.implementation.pandas.operators.window.tests.simple_moving_average.data import diff_sampling
from temporal_feature_processor.implementation.pandas.operators.window.tests.simple_moving_average.data import many_events_per_day
from temporal_feature_processor.implementation.pandas.operators.window.tests.simple_moving_average.data import many_features
from temporal_feature_processor.implementation.pandas.operators.window.tests.simple_moving_average.data import no_index
from temporal_feature_processor.implementation.pandas.operators.window.tests.simple_moving_average.data import same_sampling


class SimpleMovingAverageOperatorTest(absltest.TestCase):

  def test_no_index(self) -> None:
    """Test no specified index in the input data."""
    operator = simple_moving_average.PandasSimpleMovingAverageOperator(
        window_length="7d")
    output = operator(no_index.INPUT, no_index.SAMPLING)
    pd.testing.assert_frame_equal(no_index.OUTPUT, output["output"])
    self.assertTrue(no_index.OUTPUT.equals(output["output"]))

  def test_same_sampling(self) -> None:
    """Test same sampling for all index values."""
    operator = simple_moving_average.PandasSimpleMovingAverageOperator(
        window_length="7d")
    output = operator(same_sampling.INPUT, same_sampling.SAMPLING)
    pd.testing.assert_frame_equal(same_sampling.OUTPUT, output["output"])
    self.assertTrue(same_sampling.OUTPUT.equals(output["output"]))

  def test_diff_sampling(self) -> None:
    """Test different sampling for each index value."""
    operator = simple_moving_average.PandasSimpleMovingAverageOperator(
        window_length="7d")
    output = operator(diff_sampling.INPUT, diff_sampling.SAMPLING)
    pd.testing.assert_frame_equal(diff_sampling.OUTPUT, output["output"])
    self.assertTrue(diff_sampling.OUTPUT.equals(output["output"]))

  def test_many_events_per_day(self) -> None:
    """Test several events occuring per day."""
    operator = simple_moving_average.PandasSimpleMovingAverageOperator(
        window_length="7d")
    output = operator(many_events_per_day.INPUT, many_events_per_day.SAMPLING)
    pd.testing.assert_frame_equal(many_events_per_day.OUTPUT, output["output"])
    self.assertTrue(many_events_per_day.OUTPUT.equals(output["output"]))

  def test_many_features(self) -> None:
    """Test several events occuring per day."""
    operator = simple_moving_average.PandasSimpleMovingAverageOperator(
        window_length="7d")
    output = operator(many_features.INPUT, many_features.SAMPLING)
    pd.testing.assert_frame_equal(many_features.OUTPUT, output["output"])
    self.assertTrue(many_features.OUTPUT.equals(output["output"]))


if __name__ == "__main__":
  absltest.main()
