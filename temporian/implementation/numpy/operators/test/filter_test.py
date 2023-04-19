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

from temporian.core.operators.filter import FilterOperator
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.operators.filter_event import (
    FilterNumpyImplementation,
)


class FilterOperatorTest(absltest.TestCase):
    """Filter operator test."""

    def test_simple_filter(self) -> None:
        """Test correct filter operator for without MultiIndex and only one
        feature."""

        event_df = pd.DataFrame(
            [
                [1.0, 10.0],
                [2.0, np.nan],
                [3.0, 12.0],
                [4.0, 13.0],
                [5.0, 14.0],
                [6.0, 15.0],
            ],
            columns=["timestamp", "sales"],
        )

        event_data = NumpyEvent.from_dataframe(event_df)
        event = event_data.schema()

        condition_df = pd.DataFrame(
            [
                [1.0, True],
                [2.0, True],
                [3.0, True],
                [4.0, False],
                [5.0, False],
                [6.0, False],
            ],
            columns=["timestamp", "low_sales"],
        )

        condition_data = NumpyEvent.from_dataframe(condition_df)
        condition = condition_data.schema()
        condition._sampling = event._sampling

        operator = FilterOperator(event=event, condition=condition)
        impl = FilterNumpyImplementation(operator)
        filtered_event = impl.call(event=event_data, condition=condition_data)[
            "event"
        ]

        expected_df = pd.DataFrame(
            [
                [1.0, 10.0],
                [2.0, np.nan],
                [3.0, 12.0],
            ],
            columns=["timestamp", "sales_filtered_by_low_sales"],
        )

        expected_event = NumpyEvent.from_dataframe(expected_df)

        self.assertEqual(filtered_event, expected_event)

    def test_simple_filter_two_features(self) -> None:
        """Test correct filter operator for without MultiIndex and two features."""

        event_df = pd.DataFrame(
            [
                [1.0, 10.0, "A"],
                [2.0, np.nan, "B"],
                [3.0, 12.0, "C"],
                [4.0, 13.0, "D"],
                [5.0, 14.0, "E"],
                [6.0, 15.0, "F"],
            ],
            columns=["timestamp", "sales", "product"],
        )

        event_data = NumpyEvent.from_dataframe(event_df)
        event = event_data.schema()

        condition_df = pd.DataFrame(
            [
                [1.0, True],
                [2.0, True],
                [3.0, True],
                [4.0, False],
                [5.0, False],
                [6.0, False],
            ],
            columns=["timestamp", "low_sales"],
        )

        condition_data = NumpyEvent.from_dataframe(condition_df)
        condition = condition_data.schema()
        condition._sampling = event._sampling

        operator = FilterOperator(event=event, condition=condition)
        impl = FilterNumpyImplementation(operator)
        filtered_event = impl.call(event=event_data, condition=condition_data)[
            "event"
        ]

        expected_df = pd.DataFrame(
            [
                [1.0, 10.0, "A"],
                [2.0, np.nan, "B"],
                [3.0, 12.0, "C"],
            ],
            columns=[
                "timestamp",
                "sales_filtered_by_low_sales",
                "product_filtered_by_low_sales",
            ],
        )

        expected_event = NumpyEvent.from_dataframe(expected_df)

        self.assertEqual(filtered_event, expected_event)

    def test_filter_with_multiple_index(self) -> None:
        """Test correct filter operator with multpile index."""

        event_df = pd.DataFrame(
            [
                [1.0, 10.0, "A"],
                [2.0, np.nan, "A"],
                [3.0, 12.0, "B"],
                [4.0, 13.0, "B"],
                [5.0, 14.0, "C"],
                [6.0, 15.0, "C"],
            ],
            columns=["timestamp", "sales", "product"],
        )

        event_data = NumpyEvent.from_dataframe(
            event_df, index_names=["product"]
        )
        event = event_data.schema()

        # must have same index for filtering
        condition_df = pd.DataFrame(
            [
                [1.0, True, "A"],
                [2.0, True, "A"],
                [3.0, True, "B"],
                [4.0, False, "B"],
                [5.0, False, "C"],
                [6.0, False, "C"],
            ],
            columns=["timestamp", "low_sales", "product"],
        )

        condition_data = NumpyEvent.from_dataframe(
            condition_df, index_names=["product"]
        )
        condition = condition_data.schema()
        condition._sampling = event._sampling

        operator = FilterOperator(event=event, condition=condition)
        impl = FilterNumpyImplementation(operator)
        filtered_event = impl.call(event=event_data, condition=condition_data)[
            "event"
        ]

        expected_df = pd.DataFrame(
            [
                ["A", 1.0, 10.0],
                ["A", 2.0, np.nan],
                ["B", 3.0, 12.0],
            ],
            columns=[
                "product",
                "timestamp",
                "sales_filtered_by_low_sales",
            ],
        )

        expected_event = NumpyEvent.from_dataframe(
            expected_df, index_names=["product"]
        )

        self.assertEqual(filtered_event, expected_event)


if __name__ == "__main__":
    absltest.main()
