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

from temporian.implementation.numpy.operators import lag
from temporian.implementation.numpy.operators.tests.lag.test_data import (
    correct_duration_lag,
)
from temporian.implementation.numpy.operators.tests.lag.test_data import (
    correct_period_lag,
)


class SumOperatorTest(absltest.TestCase):
    def setUp(self) -> None:
        self.duration_operator = lag.NumpyLagOperator(duration=2)
        self.period_operator = lag.NumpyLagOperator(period=2)

    def test_correct_duration_lag(self) -> None:
        operator_output = self.duration_operator(
            correct_duration_lag.INPUT,
        )

        self.assertEqual(
            True,
            correct_duration_lag.OUTPUT == operator_output["event"],
        )

    def test_correct_period_lag(self) -> None:
        operator_output = self.period_operator(
            correct_period_lag.INPUT,
        )

        self.assertEqual(
            True,
            correct_period_lag.OUTPUT == operator_output["event"],
        )


if __name__ == "__main__":
    absltest.main()
