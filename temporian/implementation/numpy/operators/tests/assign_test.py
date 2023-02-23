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
from temporian.core.operators.assign import AssignOperator

from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling
from temporian.implementation.numpy.operators.assign import (
    AssignNumpyImplementation,
)


class AssignOperatorTest(absltest.TestCase):
    def setUp(self) -> None:
        self.sampling = Sampling(["user_id"])
        self.event_left = Event(
            [Feature("sales", float)],
            sampling=self.sampling,
            creator=None,
        )

        self.event_right = Event(
            [Feature("costs", float)],
            sampling=self.sampling,
            creator=None,
        )

    def test_right_repeated_timestamps(self) -> None:
        numpy_sampling_left = NumpySampling(
            names=["user_id"],
            data={
                (151591562,): np.array(
                    [
                        "2022-02-05",
                        "2022-02-06",
                        "2022-02-07",
                    ],
                    dtype="datetime64",
                ),
                (191562515,): np.array(["2022-02-05"], dtype="datetime64"),
            },
        )

        numpy_sampling_right = NumpySampling(
            names=["user_id"],
            data={
                (151591562,): np.array(
                    ["2022-02-08", "2022-02-09", "2022-02-09"],
                    # 2022-02-09 is repeated here should be a problem
                    dtype="datetime64",
                ),
                (191562515,): np.array(["2022-02-08"], dtype="datetime64"),
            },
        )

        numpy_left_event = NumpyEvent(
            data={
                (151591562,): [
                    NumpyFeature(
                        name="sales",
                        data=np.array([0.0, 0.0, 0.0]),
                    ),
                ],
                (191562515,): [
                    NumpyFeature(
                        name="sales",
                        data=np.array([0.0]),
                    ),
                ],
            },
            sampling=numpy_sampling_left,
        )

        numpy_right_event = NumpyEvent(
            data={
                (151591562,): [
                    NumpyFeature(
                        name="costs",
                        data=np.array([0.0, 0.0, 0.0]),
                    ),
                ],
                (191562515,): [
                    NumpyFeature(
                        name="costs",
                        data=np.array([0.0]),
                    ),
                ],
            },
            sampling=numpy_sampling_right,
        )

        operator = AssignOperator(
            left_event=self.event_left, right_event=self.event_right
        )
        assign_implementation = AssignNumpyImplementation(
            operator=operator,
        )

        # output_event = assign_implementation(left_event=numpy_left_event, right_event=numpy_right_event))

        self.assertRaisesRegex(
            ValueError,
            "right sequence cannot have repeated timestamps in the same index.",
            assign_implementation,
            numpy_left_event,
            numpy_right_event,
        )

    def test_left_repeated_timestamps(self) -> None:
        numpy_sampling_left = NumpySampling(
            names=["user_id"],
            data={
                (151591562,): np.array(
                    [
                        "2022-02-05",
                        "2022-02-05",  # 2022-02-05 is repeated here should not be a problem
                        "2022-02-07",
                    ],
                    dtype="datetime64",
                ),
                (191562515,): np.array(["2022-02-05"], dtype="datetime64"),
            },
        )

        numpy_sampling_right = NumpySampling(
            names=["user_id"],
            data={
                (151591562,): np.array(
                    ["2022-02-05", "2022-02-07", "2022-02-09"],
                    dtype="datetime64",
                ),
                (191562515,): np.array(["2022-02-05"], dtype="datetime64"),
            },
        )

        numpy_left_event = NumpyEvent(
            data={
                (151591562,): [
                    NumpyFeature(
                        name="sales",
                        data=np.array([10.0, 20.0, 30.0]),
                    ),
                ],
                (191562515,): [
                    NumpyFeature(
                        name="sales",
                        data=np.array([40.0]),
                    ),
                ],
            },
            sampling=numpy_sampling_left,
        )

        numpy_right_event = NumpyEvent(
            data={
                (151591562,): [
                    NumpyFeature(
                        name="costs",
                        data=np.array([-10.0, -20.0, 0.0]),
                    ),
                ],
                (191562515,): [
                    NumpyFeature(
                        name="costs",
                        data=np.array([0.0]),
                    ),
                ],
            },
            sampling=numpy_sampling_right,
        )

        expected_numpy_event = NumpyEvent(
            data={
                (151591562,): [
                    NumpyFeature(
                        name="sales",
                        data=np.array([10.0, 20.0, 30.0]),
                    ),
                    NumpyFeature(
                        name="costs",
                        data=np.array(
                            [-10.0, -10.0, -20.0]
                        ),  # -10.0 is repeated here
                        # because the timestamp is repeated
                    ),
                ],
                (191562515,): [
                    NumpyFeature(
                        name="sales",
                        data=np.array([40.0]),
                    ),
                    NumpyFeature(
                        name="costs",
                        data=np.array([0.0]),
                    ),
                ],
            },
            sampling=numpy_sampling_left,
        )

        operator = AssignOperator(
            left_event=self.event_left, right_event=self.event_right
        )
        assign_implementation = AssignNumpyImplementation(
            operator=operator,
        )

        output_event = assign_implementation(
            left_event=numpy_left_event, right_event=numpy_right_event
        )

        self.assertEqual(
            True,
            output_event["event"] == expected_numpy_event,
        )

    def test_different_index(self) -> None:
        numpy_sampling_left_event = NumpySampling(
            names=["store_id"],
            data={
                ("A",): np.array(
                    ["2022-02-05", "2022-02-06", "2022-02-07"],
                    dtype="datetime64",
                ),
                ("B",): np.array(
                    ["2022-02-05", "2022-02-06"], dtype="datetime64"
                ),
            },
        )

        numpy_sampling_right_event = NumpySampling(
            names=["product_id"],
            data={
                (1,): np.array(
                    ["2022-02-05", "2022-02-06", "2022-02-07"],
                    dtype="datetime64",
                ),
                (2,): np.array(
                    ["2022-02-05", "2022-02-06"], dtype="datetime64"
                ),
            },
        )

        numpy_left_event = NumpyEvent(
            data={
                ("A",): [
                    NumpyFeature(
                        name="sales",
                        data=np.array([14, 15, 16]),
                    ),
                ],
                ("B",): [
                    NumpyFeature(
                        name="sales",
                        data=np.array([10, 11]),
                    ),
                ],
            },
            sampling=numpy_sampling_left_event,
        )

        numpy_right_event = NumpyEvent(
            data={
                (1,): [
                    NumpyFeature(
                        name="costs",
                        data=np.array([-14, -15, -16]),
                    ),
                ],
                (2,): [
                    NumpyFeature(
                        name="costs",
                        data=np.array([-10, -11]),
                    ),
                ],
            },
            sampling=numpy_sampling_right_event,
        )

        left_event = Event([Feature("sales")], Sampling(["store_id"]))
        right_event = Event([Feature("costs")], Sampling(["product_id"]))

        operator = AssignOperator(
            left_event=left_event, right_event=right_event
        )
        assign_implementation = AssignNumpyImplementation(
            operator=operator,
        )

        self.assertRaisesRegex(
            ValueError,
            "Assign sequences must have the same index names.",
            assign_implementation,
            numpy_left_event,
            numpy_right_event,
        )

    def test_with_idx_more_timestamps(self) -> None:
        """Tests the correct output when the right event has more timestamps than the left
        event, for any index value. Both input events are indexed.
        """
        numpy_sampling_left = NumpySampling(
            names=["user_id"],
            data={
                (666964,): np.array(
                    ["2022-02-05"],
                    dtype="datetime64",
                ),
                (372306,): np.array(["2022-02-06"], dtype="datetime64"),
            },
        )

        numpy_sampling_right = NumpySampling(
            names=["user_id"],
            data={
                (666964,): np.array(
                    ["2022-02-05"],
                    dtype="datetime64",
                ),
                (372306,): np.array(
                    ["2022-02-06", "2022-02-07"], dtype="datetime64"
                ),
            },
        )

        numpy_left_event = NumpyEvent(
            data={
                (666964,): [
                    NumpyFeature(
                        name="sales",
                        data=np.array([0.0]),
                    ),
                ],
                (372306,): [
                    NumpyFeature(
                        name="sales",
                        data=np.array([1160.0]),
                    ),
                ],
            },
            sampling=numpy_sampling_left,
        )

        numpy_right_event = NumpyEvent(
            data={
                (666964,): [
                    NumpyFeature(
                        name="costs",
                        data=np.array([0.0]),
                    ),
                ],
                (372306,): [
                    NumpyFeature(
                        name="costs",
                        data=np.array([508.0, 573.0]),
                    ),
                ],
            },
            sampling=numpy_sampling_right,
        )

        expected_output = NumpyEvent(
            data={
                (666964,): [
                    NumpyFeature(
                        name="sales",
                        data=np.array([0.0]),
                    ),
                    NumpyFeature(
                        name="costs",
                        data=np.array([0.0]),
                    ),
                ],
                (372306,): [
                    NumpyFeature(
                        name="sales",
                        data=np.array([1160.0]),
                    ),
                    NumpyFeature(
                        name="costs",
                        data=np.array([508.0]),
                    ),
                ],
            },
            sampling=numpy_sampling_left,
        )

        operator = AssignOperator(
            left_event=self.event_left, right_event=self.event_right
        )
        assign_implementation = AssignNumpyImplementation(
            operator=operator,
        )

        output_event = assign_implementation(
            left_event=numpy_left_event, right_event=numpy_right_event
        )

        self.assertEqual(
            True,
            output_event["event"] == expected_output,
        )

    def test_with_idx_same_timestamps(self) -> None:
        """Tests the correct output when the right event has same timestamps than the left
        event, for any index value. Both input events are indexed.
        """
        numpy_sampling_left = NumpySampling(
            names=["user_id"],
            data={
                (666964,): np.array(
                    ["2022-02-05", "2022-02-06", "2022-02-07"],
                    dtype="datetime64",
                ),
                (372306,): np.array(["2022-02-06"], dtype="datetime64"),
            },
        )

        numpy_sampling_right = NumpySampling(
            names=["user_id"],
            data={
                (666964,): np.array(
                    ["2022-02-05", "2022-02-06", "2022-02-07"],
                    dtype="datetime64",
                ),
                (372306,): np.array(["2022-02-06"], dtype="datetime64"),
            },
        )

        numpy_left_event = NumpyEvent(
            data={
                (666964,): [
                    NumpyFeature(
                        name="sales",
                        data=np.array([0.0, 100.0, 200.0]),
                    ),
                ],
                (372306,): [
                    NumpyFeature(
                        name="sales",
                        data=np.array([1160.0]),
                    ),
                ],
            },
            sampling=numpy_sampling_left,
        )

        numpy_right_event = NumpyEvent(
            data={
                (666964,): [
                    NumpyFeature(
                        name="costs",
                        data=np.array([0.0, 250.0, 500.0]),
                    ),
                ],
                (372306,): [
                    NumpyFeature(
                        name="costs",
                        data=np.array([508.0]),
                    ),
                ],
            },
            sampling=numpy_sampling_right,
        )

        expected_output = NumpyEvent(
            data={
                (666964,): [
                    NumpyFeature(
                        name="sales",
                        data=np.array([0.0, 100.0, 200.0]),
                    ),
                    NumpyFeature(
                        name="costs",
                        data=np.array([0.0, 250.0, 500.0]),
                    ),
                ],
                (372306,): [
                    NumpyFeature(
                        name="sales",
                        data=np.array([1160.0]),
                    ),
                    NumpyFeature(
                        name="costs",
                        data=np.array([508.0]),
                    ),
                ],
            },
            sampling=numpy_sampling_left,
        )

        operator = AssignOperator(
            left_event=self.event_left, right_event=self.event_right
        )
        assign_implementation = AssignNumpyImplementation(
            operator=operator,
        )

        output_event = assign_implementation(
            left_event=numpy_left_event, right_event=numpy_right_event
        )

        self.assertEqual(
            True,
            output_event["event"] == expected_output,
        )

    def test_less_right_indexes(self) -> None:
        numpy_sampling_left = NumpySampling(
            names=["user_id"],
            data={
                ("A",): np.array(
                    ["2022-02-05", "2022-02-06", "2022-02-07"],
                    dtype="datetime64",
                ),
                ("B",): np.array(
                    ["2022-02-05", "2022-02-06"], dtype="datetime64"
                ),
                ("C",): np.array(
                    ["2022-02-05", "2022-02-06"], dtype="datetime64"
                ),
            },
        )

        numpy_sampling_right = NumpySampling(
            names=["user_id"],
            data={
                ("A",): np.array(
                    ["2022-02-05", "2022-02-06", "2022-02-07"],
                    dtype="datetime64",
                ),
                # Missing B index that will be broadcasted
                ("C",): np.array(
                    ["2022-02-05", "2022-02-06"], dtype="datetime64"
                ),
            },
        )

        numpy_left_event = NumpyEvent(
            data={
                ("A",): [
                    NumpyFeature(
                        name="sales",
                        data=np.array([14, 15, 16]),
                    ),
                ],
                ("B",): [
                    NumpyFeature(
                        name="sales",
                        data=np.array([10, 11]),
                    ),
                ],
                ("C",): [
                    NumpyFeature(
                        name="sales",
                        data=np.array([9, 10]),
                    ),
                ],
            },
            sampling=numpy_sampling_left,
        )

        numpy_right_event = NumpyEvent(
            data={
                ("A",): [
                    NumpyFeature(
                        name="costs",
                        data=np.array([-14, -15, -16]),
                    ),
                ],
                ("C",): [
                    NumpyFeature(
                        name="costs",
                        data=np.array([-9, -10]),
                    ),
                ],
            },
            sampling=numpy_sampling_right,
        )

        expected_output = NumpyEvent(
            data={
                ("A",): [
                    NumpyFeature(
                        name="sales",
                        data=np.array([14, 15, 16]),
                    ),
                    NumpyFeature(
                        name="costs",
                        data=np.array([-14, -15, -16]),
                    ),
                ],
                ("B",): [
                    NumpyFeature(
                        name="sales",
                        data=np.array([10, 11]),
                    ),
                    NumpyFeature(
                        name="costs",
                        data=np.array([np.nan, np.nan]),  # broadcasted feature
                    ),
                ],
                ("C",): [
                    NumpyFeature(
                        name="sales",
                        data=np.array([9, 10]),
                    ),
                    NumpyFeature(
                        name="costs",
                        data=np.array([-9, -10]),
                    ),
                ],
            },
            sampling=numpy_sampling_left,
        )

        operator = AssignOperator(
            left_event=self.event_left, right_event=self.event_right
        )
        assign_implementation = AssignNumpyImplementation(
            operator=operator,
        )

        output_event = assign_implementation(
            left_event=numpy_left_event, right_event=numpy_right_event
        )

        self.assertEqual(
            True,
            output_event["event"] == expected_output,
        )

    def test_complete_timestamps(self) -> None:
        """Tests a correct output in a complete timestamps scenario. both samplings will have different timestamps and
        in different order."""
        numpy_sampling_left = NumpySampling(
            names=["user_id"],
            data={
                (666964,): np.array(
                    [
                        "2022-02-01",
                        "2022-02-02",
                        "2022-02-03",
                        "2022-02-04",
                        "2022-02-05",
                    ],
                    dtype="datetime64",
                ),
            },
        )

        numpy_sampling_right = NumpySampling(
            names=["user_id"],
            data={
                (666964,): np.array(
                    [  # missing timestamps from sampling_1 (2022-02-01, 2022-02-03, 2022-02-04)
                        "2022-01-31",  # not in sampling_1
                        "2022-02-02",  # its on sampling_1 but in different index
                        "2022-02-05",  # its on sampling_1 but in different index
                        "2022-02-07",  # not in sampling_1
                    ],
                    dtype="datetime64",
                ),
            },
        )

        numpy_left_event = NumpyEvent(
            data={
                (666964,): [
                    NumpyFeature(
                        name="sales",
                        data=np.array([10, 11, 12, 13, 14]),
                    ),
                ],
            },
            sampling=numpy_sampling_left,
        )

        numpy_right_event = NumpyEvent(
            data={
                (666964,): [
                    NumpyFeature(
                        name="costs",
                        data=np.array([1, 2, 3, 4]),
                    ),
                ],
            },
            sampling=numpy_sampling_right,
        )

        expected_output = NumpyEvent(
            data={
                (666964,): [
                    NumpyFeature(
                        name="sales",
                        data=np.array([10, 11, 12, 13, 14]),
                    ),
                    NumpyFeature(
                        name="costs",
                        data=np.array(
                            [np.nan, 2, np.nan, np.nan, 3]
                        ),  # np.nan on missing timestamps
                        # 1 and 2 value on different index
                        # missing 3 value as its timestamp is not in sampling_1
                    ),
                ],
            },
            sampling=numpy_sampling_left,
        )

        operator = AssignOperator(
            left_event=self.event_left, right_event=self.event_right
        )
        assign_implementation = AssignNumpyImplementation(
            operator=operator,
        )

        output_event = assign_implementation(
            left_event=numpy_left_event, right_event=numpy_right_event
        )

        self.assertEqual(
            True,
            output_event["event"] == expected_output,
        )


if __name__ == "__main__":
    absltest.main()
