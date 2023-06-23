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
import pandas as pd

from temporian.core.data.node import input_node
from temporian.core.operators.calendar.day_of_month import (
    CalendarDayOfMonthOperator,
)
from temporian.io.pandas import from_pandas
from temporian.implementation.numpy.data.io import event_set

from temporian.implementation.numpy.operators.calendar.day_of_month import (
    CalendarDayOfMonthNumpyImplementation,
)
from temporian.implementation.numpy.operators.test.test_util import (
    assertEqualEventSet,
    testOperatorAndImp,
)


class CalendarDayOfMonthNumpyImplementationTest(absltest.TestCase):
    """Test numpy implementation of calendar_day_of_month operator."""

    def test_no_index(self) -> None:
        """Test calendar day of month operator with flat node."""
        input_evset = from_pandas(
            pd.DataFrame(
                data=[
                    [pd.to_datetime("1970-01-01 00:00:00", utc=True)],
                    [pd.to_datetime("2023-03-14 00:00:01", utc=True)],
                    [pd.to_datetime("2023-03-14 23:59:59", utc=True)],
                    [pd.to_datetime("2023-03-15 12:00:00", utc=True)],
                ],
                columns=["timestamp"],
            ),
        )
        output_evset = event_set(
            timestamps=input_evset.get_arbitrary_index_data().timestamps,
            features={
                "calendar_day_of_month": np.array([1, 14, 14, 15]).astype(
                    np.int32
                ),
            },
            is_unix_timestamp=True,
        )
        operator = CalendarDayOfMonthOperator(input_evset.node())
        impl = CalendarDayOfMonthNumpyImplementation(operator)
        testOperatorAndImp(self, operator, impl)
        output = impl.call(sampling=input_evset)["output"]

        assertEqualEventSet(self, output, output_evset)
        self.assertTrue(
            output.get_arbitrary_index_data().features[0].dtype == np.int32
        )

    def test_with_index(self) -> None:
        """Test calendar day of month operator with indexed node."""
        input_evset = from_pandas(
            pd.DataFrame(
                data=[
                    [pd.to_datetime("1970-01-01 00:00:00", utc=True), 1],
                    [pd.to_datetime("2023-03-14 00:00:01", utc=True), 1],
                    [pd.to_datetime("2023-03-14 00:00:01", utc=True), 2],
                    [pd.to_datetime("2023-03-14 23:59:59", utc=True), 1],
                    [pd.to_datetime("2023-03-15 12:00:00", utc=True), 2],
                ],
                columns=["timestamp", "id"],
            ),
            indexes=["id"],
        )

        output_evset = event_set(
            timestamps=np.concatenate(
                [
                    input_evset.data[(1,)].timestamps,
                    input_evset.data[(2,)].timestamps,
                ]
            ),
            features={
                "calendar_day_of_month": np.array([1, 14, 14, 14, 15]).astype(
                    np.int32
                ),
                "id": [1, 1, 1, 2, 2],
            },
            indexes=["id"],
            is_unix_timestamp=True,
        )
        operator = CalendarDayOfMonthOperator(input_evset.node())
        operator.outputs["output"].check_same_sampling(input_evset.node())

        impl = CalendarDayOfMonthNumpyImplementation(operator)
        output = impl.call(sampling=input_evset)["output"]

        assertEqualEventSet(self, output, output_evset)
        self.assertTrue(
            output.get_arbitrary_index_data().features[0].dtype == np.int32
        )

    # TODO: move this test to core operators' test suite when created
    # since its testing BaseCalendarOperator's logic, not
    # CalendarDayOfMonthNumpyImplementation's
    def test_invalid_sampling(self) -> None:
        """
        Test calendar operator with a non-utc timestamp
        sampling.
        """

        with self.assertRaises(ValueError):
            CalendarDayOfMonthOperator(
                input_node(features=[], is_unix_timestamp=False)
            )


if __name__ == "__main__":
    absltest.main()
