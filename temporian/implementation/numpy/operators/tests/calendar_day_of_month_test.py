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

from temporian.core.data.event import Event
from temporian.core.data.event import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators.calendar_day_of_month import (
    CalendarDayOfMonthOperator,
)
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling
from temporian.implementation.numpy.operators.calendar_day_of_month import (
    CalendarDayOfMonthNumpyImplementation,
)


class CalendarDayNumpyImplementationTest(absltest.TestCase):
    """Test numpy implementation of calendar_day_of_month operator."""

    def test_day(self) -> None:
        """Test calendar day operator."""
        input_sampling_data = NumpySampling(
            index=[],
            data={
                (): np.array(
                    [
                        0,  # 1970-01-01
                        1678895040,  # 2023-03-15
                        1678981440,  # 2023-03-16
                        1679067840,  # 2023-03-17
                    ],
                    dtype=np.float64,
                ),
            },
        )

        # we don't care about the actual feature values, just the sampling
        input_event_data = NumpyEvent(
            data={
                (): [
                    NumpyFeature(
                        name="feature",
                        data=np.array([0, 0, 0, 0]),
                    ),
                ],
            },
            sampling=input_sampling_data,
        )

        input_sampling = Sampling([])

        input_event = Event(
            [Feature("feature", float)],
            sampling=input_sampling,
            creator=None,
        )

        output_event_data = NumpyEvent(
            data={
                (): [
                    NumpyFeature(
                        name="calendar_day_of_month",
                        data=np.array([1, 15, 16, 17]),
                    ),
                ],
            },
            sampling=input_sampling_data,
        )

        operator = CalendarDayOfMonthOperator(input_event)
        impl = CalendarDayOfMonthNumpyImplementation(operator)

        output = impl(input_event_data)

        self.assertTrue(output_event_data == output["event"])


if __name__ == "__main__":
    absltest.main()
