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

import pandas as pd
import numpy as np

from temporian.core.operators.simple_moving_average import SimpleMovingAverage
from temporian.implementation.numpy.operators.window.simple_moving_average import (
    SimpleMovingAverageNumpyImplementation,
)
from temporian.implementation.numpy.data.event import NumpyEvent, NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling
from temporian.core.data import event as event_lib
from temporian.core.data import feature as feature_lib
from temporian.core.data import dtype as dtype_lib


class SimpleMovingAverageOperatorTest(absltest.TestCase):
    def setUp(self):
        pass

    # TODO: Add more tests (index, sampling, nan, etc), notably the ones in the
    # pandas backend.

    def test_no_index(self):
        # TODO: Simplify test definition with "pd_to_event" when available.
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
        output = instance(event=input_data)
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


if __name__ == "__main__":
    absltest.main()
