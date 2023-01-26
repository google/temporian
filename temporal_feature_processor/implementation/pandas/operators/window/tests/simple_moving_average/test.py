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

from temporal_feature_processor.implementation.pandas.operators.window import \
    simple_moving_average
from temporal_feature_processor.implementation.pandas.operators.window.tests.simple_moving_average.data import (
    no_index, same_sampling)


class SimpleMovingAverageOperatorTest(absltest.TestCase):

  def test_no_index(self) -> None:
    """Test simple moving average operator with no specified index in the input
    data."""
    operator = simple_moving_average.PandasSimpleMovingAverageOperator(
        window_length="7d")
    output = operator(no_index.INPUT, no_index.SAMPLING)
    pd.testing.assert_frame_equal(no_index.OUTPUT, output["output"])
    self.assertTrue(no_index.OUTPUT.equals(output["output"]))

  def test_same_sampling(self) -> None:
    """Test simple moving average operator with same sampling for all index
    values."""
    operator = simple_moving_average.PandasSimpleMovingAverageOperator(
        window_length="7d")
    output = operator(same_sampling.INPUT, same_sampling.SAMPLING)
    pd.testing.assert_frame_equal(same_sampling.OUTPUT, output["output"])
    self.assertTrue(same_sampling.OUTPUT.equals(output["output"]))


if __name__ == "__main__":
  absltest.main()
