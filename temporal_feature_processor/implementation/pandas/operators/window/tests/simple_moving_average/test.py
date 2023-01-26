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

from temporal_feature_processor.implementation.pandas.operators.window import \
    simple_moving_average
from temporal_feature_processor.implementation.pandas.operators.window.tests.simple_moving_average.data import \
    no_index


class SimpleMovingAverageOperatorTest(absltest.TestCase):

  def test_no_index(self) -> None:
    """Test simple moving average operator with no specified index in the input
    data."""
    operator = simple_moving_average.PandasSimpleMovingAverageOperator(
        window_length="7d")
    output = operator(no_index.INPUT, no_index.SAMPLING)
    self.assertTrue(no_index.OUTPUT.equals(output["output"]))


if __name__ == "__main__":
  absltest.main()
