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
import math

import numpy as np

from temporian.core.operators.simple_moving_average import SimpleMovingAverage
from temporian.implementation.numpy.operators.simple_moving_average import (
    SimpleMovingAverageNumpyImplementation,
)
from temporian.implementation.numpy.data.event import NumpyEvent, NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling
from temporian.core.data import event as event_lib
from temporian.core.data import feature as feature_lib
from temporian.core.data import dtype as dtype_lib
from temporian.implementation.numpy.evaluator import run_with_check


class SimpleMovingAverageOperatorTest(absltest.TestCase):
    def setUp(self):
        pass

    # TODO: Import tests from pandas backend.
    # TODO: Simplify tests with "pd_to_event".

    def test_flat(self):
        """A simple time sequence."""

        op = SimpleMovingAverage(
            event=event_lib.input_event(
                [
                    feature_lib.Feature(name="a", dtype=dtype_lib.FLOAT64),
                    feature_lib.Feature(name="b", dtype=dtype_lib.FLOAT64),
                ]
            ),
            window_length=5,
            sampling=None,
        )
        self.assertEqual(op.list_matching_io_samplings(), [("event", "event")])
        instance = SimpleMovingAverageNumpyImplementation(op)

        input_data = NumpyEvent(
            data={
                (): [
                    NumpyFeature(
                        name="a",
                        data=np.array([10.0, 11.0, 12.0, 13.0, 14.0]),
                    ),
                    NumpyFeature(
                        name="b",
                        data=np.array([20, 21, 22, 23, 24]),
                    ),
                ]
            },
            sampling=NumpySampling(
                index=[],
                data={(): np.array([1, 2, 3, 5, 20], dtype=np.float64)},
            ),
        )
        output = run_with_check(op, instance, {"event": input_data})
        expected_output = NumpyEvent(
            data={
                (): [
                    NumpyFeature(
                        name="sma_a",
                        data=np.array([10.0, 10.5, 11.0, 11.5, 14.0]),
                    ),
                    NumpyFeature(
                        name="sma_b",
                        data=np.array([20.0, 20.5, 21.0, 21.5, 24.0]),
                    ),
                ]
            },
            sampling=NumpySampling(
                index=[],
                data={(): np.array([1, 2, 3, 5, 20], dtype=np.float64)},
            ),
        )
        self.assertEqual(repr(output), repr({"event": expected_output}))

    def test_with_index(self):
        """Indexed time sequences."""

        op = SimpleMovingAverage(
            event=event_lib.input_event(
                [feature_lib.Feature(name="a", dtype=dtype_lib.FLOAT64)],
                index=["x", "y"],
            ),
            window_length=5,
            sampling=None,
        )
        self.assertEqual(op.list_matching_io_samplings(), [("event", "event")])
        instance = SimpleMovingAverageNumpyImplementation(op)

        input_data = NumpyEvent(
            data={
                ("X1", "Y1"): [
                    NumpyFeature(
                        name="a",
                        data=np.array([10.0, 11.0, 12.0]),
                    )
                ],
                ("X2", "Y1"): [
                    NumpyFeature(
                        name="a",
                        data=np.array([13.0, 14.0, 15.0]),
                    )
                ],
                ("X2", "Y2"): [
                    NumpyFeature(
                        name="a",
                        data=np.array([16.0, 17.0, 18.0]),
                    )
                ],
            },
            sampling=NumpySampling(
                index=["x", "y"],
                data={
                    ("X1", "Y1"): np.array([1, 2, 3], dtype=np.float64),
                    ("X2", "Y1"): np.array([1.1, 2.1, 3.1], dtype=np.float64),
                    ("X2", "Y2"): np.array([1.2, 2.2, 3.2], dtype=np.float64),
                },
            ),
        )
        output = run_with_check(op, instance, {"event": input_data})

        expected_output = NumpyEvent(
            data={
                ("X1", "Y1"): [
                    NumpyFeature(
                        name="sma_a",
                        data=np.array([10.0, 10.5, 11.0]),
                    )
                ],
                ("X2", "Y1"): [
                    NumpyFeature(
                        name="sma_a",
                        data=np.array([13.0, 13.5, 14.0]),
                    )
                ],
                ("X2", "Y2"): [
                    NumpyFeature(
                        name="sma_a",
                        data=np.array([16.0, 16.5, 17.0]),
                    )
                ],
            },
            sampling=NumpySampling(
                index=["x", "y"],
                data={
                    ("X1", "Y1"): np.array([1.0, 2.0, 3.0], dtype=np.float64),
                    ("X2", "Y1"): np.array([1.1, 2.1, 3.1], dtype=np.float64),
                    ("X2", "Y2"): np.array([1.2, 2.2, 3.2], dtype=np.float64),
                },
            ),
        )
        self.assertEqual(repr(output), repr({"event": expected_output}))

    def test_with_sampling(self):
        """Time sequenes with user provided sampling."""

        op = SimpleMovingAverage(
            event=event_lib.input_event(
                [feature_lib.Feature(name="a", dtype=dtype_lib.FLOAT64)]
            ),
            window_length=3,
            sampling=event_lib.input_event([]),
        )
        self.assertEqual(
            op.list_matching_io_samplings(), [("sampling", "event")]
        )
        instance = SimpleMovingAverageNumpyImplementation(op)

        input_data = NumpyEvent(
            data={
                (): [
                    NumpyFeature(
                        name="a",
                        data=np.array([10.0, 11.0, 12.0, 13.0, 14.0]),
                    ),
                ]
            },
            sampling=NumpySampling(
                index=[],
                data={(): np.array([1, 2, 3, 5, 6], dtype=np.float64)},
            ),
        )

        sampling_data = NumpyEvent(
            data={(): []},
            sampling=NumpySampling(
                index=[],
                data={
                    (): np.array(
                        [-1.0, 1.0, 1.1, 3, 3.5, 6, 10], dtype=np.float64
                    )
                },
            ),
        )

        output = run_with_check(
            op, instance, {"event": input_data, "sampling": sampling_data}
        )

        expected_output = NumpyEvent(
            data={
                (): [
                    NumpyFeature(
                        name="sma_a",
                        data=np.array(
                            [math.nan, 10.0, 10.0, 11.0, 11.0, 13.0, math.nan]
                        ),
                    ),
                ]
            },
            sampling=NumpySampling(
                index=[],
                data={
                    (): np.array(
                        [-1.0, 1.0, 1.1, 3.0, 3.5, 6.0, 10.0], dtype=np.float64
                    )
                },
            ),
        )
        self.assertEqual(repr(output), repr({"event": expected_output}))

    def test_with_nan(self):
        """The input features contains nan values."""

        op = SimpleMovingAverage(
            event=event_lib.input_event(
                [feature_lib.Feature(name="a", dtype=dtype_lib.FLOAT64)]
            ),
            window_length=1,
            sampling=event_lib.input_event([]),
        )
        instance = SimpleMovingAverageNumpyImplementation(op)

        input_data = NumpyEvent(
            data={
                (): [
                    NumpyFeature(
                        name="a",
                        data=np.array([math.nan, 11.0, math.nan, 13.0, 14.0]),
                    ),
                ]
            },
            sampling=NumpySampling(
                index=[],
                data={(): np.array([1, 2, 3, 5, 6], dtype=np.float64)},
            ),
        )

        sampling_data = NumpyEvent(
            data={(): []},
            sampling=NumpySampling(
                index=[],
                data={
                    (): np.array([1, 2, 2.5, 3, 3.5, 4, 5, 6], dtype=np.float64)
                },
            ),
        )

        output = run_with_check(
            op, instance, {"event": input_data, "sampling": sampling_data}
        )

        expected_output = NumpyEvent(
            data={
                (): [
                    NumpyFeature(
                        name="sma_a",
                        data=np.array(
                            [
                                math.nan,
                                11.0,
                                11,
                                11.0,
                                math.nan,
                                math.nan,
                                13.0,
                                13.5,
                            ]
                        ),
                    ),
                ]
            },
            sampling=NumpySampling(
                index=[],
                data={
                    (): np.array([1, 2, 2.5, 3, 3.5, 4, 5, 6], dtype=np.float64)
                },
            ),
        )
        self.assertEqual(repr(output), repr({"event": expected_output}))


if __name__ == "__main__":
    absltest.main()
