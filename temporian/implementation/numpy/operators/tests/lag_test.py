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

import numpy as np

from temporian.core.data.sampling import Sampling
from temporian.core.data.event import Event
from temporian.core.data.event import Feature
from temporian.core.operators.lag import LagOperator
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling
from temporian.implementation.numpy.operators import lag


class LagOperatorTest(absltest.TestCase):
    """Lag operator test."""

    def test_correct_lag(self) -> None:
        """Test correct lag operator."""
        # DATA
        numpy_input_sampling = NumpySampling(
            names=["store_id"],
            data={("A",): np.array([1, 1.5, 3, 3.5, 4, 10, 20])},
        )

        numpy_input_event = NumpyEvent(
            data={
                ("A",): [
                    NumpyFeature(
                        name="sales",
                        data=np.array(
                            [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
                        ),
                    ),
                ],
            },
            sampling=numpy_input_sampling,
        )

        numpy_output_sampling = NumpySampling(
            names=["store_id"],
            data={("A",): np.array([3, 3.5, 5, 5.5, 6, 12, 22])},
        )

        numpy_output_event = NumpyEvent(
            data={
                ("A",): [
                    NumpyFeature(
                        name="lag_sales",
                        data=np.array(
                            [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
                        ),
                    ),
                ],
            },
            sampling=numpy_output_sampling,
        )

        event = Event(
            [Feature("sales", float)],
            sampling=Sampling(["store_id"]),
            creator=None,
        )

        operator = LagOperator(
            duration=2,
            event=event,
        )

        lag_implementation = lag.LagNumpyImplementation(operator)
        operator_output = lag_implementation(event=numpy_input_event)

        self.assertEqual(
            True,
            numpy_output_event == operator_output["event"],
        )


if __name__ == "__main__":
    absltest.main()
