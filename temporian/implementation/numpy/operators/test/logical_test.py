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

import numpy as np
import pandas as pd
from absl.testing import absltest

from temporian.core.operators.binary import (
    LogicalAndOperator,
    LogicalOrOperator,
    LogicalXorOperator,
)
from temporian.implementation.numpy.operators.binary import (
    LogicalAndNumpyImplementation,
    LogicalOrNumpyImplementation,
    LogicalXorNumpyImplementation,
)
from temporian.implementation.numpy.data.io import pd_dataframe_to_event_set


class ArithmeticNumpyImplementationTest(absltest.TestCase):
    """Test numpy implementation of all arithmetic operators:
    addition, subtraction, division and multiplication"""

    def setUp(self):
        self.evset_1 = pd_dataframe_to_event_set(
            pd.DataFrame(
                {"timestamp": [1, 2, 3, 4], "x": [True, False, True, False]}
            )
        )
        self.evset_2 = pd_dataframe_to_event_set(
            pd.DataFrame(
                {"timestamp": [1, 2, 3, 4], "x": [True, False, False, True]}
            )
        )
        self.node_1 = self.evset_1.node()
        self.node_2 = self.evset_2.node()

        # FIXME: This should not be necessary
        self.node_2._sampling = self.node_1._sampling
        for index, data in self.evset_1.data.items():
            self.evset_2[index].timestamps = data.timestamps

    def test_correct_and(self) -> None:
        """Test correct AND operator."""
        output_evset = pd_dataframe_to_event_set(
            pd.DataFrame(
                {
                    "timestamp": [1, 2, 3, 4],
                    "and_x_x": [True, False, False, False],
                }
            )
        )
        operator = LogicalAndOperator(input_1=self.node_1, input_2=self.node_2)
        operator.outputs["output"].check_same_sampling(self.node_1)
        operator.outputs["output"].check_same_sampling(self.node_2)

        operator_output = LogicalAndNumpyImplementation(operator).call(
            input_1=self.evset_1, input_2=self.evset_2
        )
        self.assertEqual(output_evset, operator_output["output"])

    def test_correct_or(self) -> None:
        """Test correct OR operator."""
        output_evset = pd_dataframe_to_event_set(
            pd.DataFrame(
                {"timestamp": [1, 2, 3, 4], "or_x_x": [True, False, True, True]}
            )
        )
        operator = LogicalOrOperator(input_1=self.node_1, input_2=self.node_2)
        operator_output = LogicalOrNumpyImplementation(operator).call(
            input_1=self.evset_1, input_2=self.evset_2
        )
        self.assertEqual(output_evset, operator_output["output"])

    def test_correct_xor(self) -> None:
        """Test correct XOR operator."""
        output_evset = pd_dataframe_to_event_set(
            pd.DataFrame(
                {
                    "timestamp": [1, 2, 3, 4],
                    "xor_x_x": [False, False, True, True],
                }
            )
        )
        operator = LogicalXorOperator(input_1=self.node_1, input_2=self.node_2)
        operator_output = LogicalXorNumpyImplementation(operator).call(
            input_1=self.evset_1, input_2=self.evset_2
        )
        self.assertEqual(output_evset, operator_output["output"])


if __name__ == "__main__":
    absltest.main()
