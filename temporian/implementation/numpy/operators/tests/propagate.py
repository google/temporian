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

from temporian.core.operators.propagate import Propagate, propagate
from temporian.implementation.numpy.operators.propagate import (
    PropagateNumpyImplementation,
)
from temporian.implementation.numpy.data.event import NumpyEvent, NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling
from temporian.core.data import event as event_lib
from temporian.core.data import feature as feature_lib
from temporian.core.data import dtype as dtype_lib
import math


class PropagateOperatorTest(absltest.TestCase):
    def setUp(self):
        pass

    def test_base(self):
        op = Propagate(
            event=event_lib.input_event(
                [
                    feature_lib.Feature(name="a", dtype=dtype_lib.FLOAT64),
                    feature_lib.Feature(name="b", dtype=dtype_lib.FLOAT64),
                ],
                index=["x"],
            ),
            add_index=event_lib.input_event(
                [
                    feature_lib.Feature(name="c", dtype=dtype_lib.STRING),
                    feature_lib.Feature(name="d", dtype=dtype_lib.STRING),
                ],
                index=["x"],
            ),
        )
        instance = PropagateNumpyImplementation(op)

        sampling = NumpySampling(
            index=["x"],
            data={
                ("X1",): np.array([0.1, 0.2, 0.3], dtype=np.float64),
                ("X2",): np.array([0.4, 0.5], dtype=np.float64),
            },
        )

        input_data = NumpyEvent(
            data={
                ("X1",): [
                    NumpyFeature(
                        name="a",
                        data=np.array([1, 2, 3]),
                    ),
                    NumpyFeature(
                        name="b",
                        data=np.array([4, 5, 6]),
                    ),
                ],
                ("X2",): [
                    NumpyFeature(
                        name="a",
                        data=np.array([7, 8]),
                    ),
                    NumpyFeature(
                        name="b",
                        data=np.array([9, 10]),
                    ),
                ],
            },
            sampling=sampling,
        )

        add_event = NumpyEvent(
            data={
                ("X1",): [
                    NumpyFeature(
                        name="c",
                        data=np.array(["C1", "C2", "C1"]),
                    ),
                    NumpyFeature(
                        name="d",
                        data=np.array(["D1", "D1", "D2"]),
                    ),
                ],
                ("X2",): [
                    NumpyFeature(
                        name="c",
                        data=np.array(["C1", "C1"]),
                    ),
                    NumpyFeature(
                        name="d",
                        data=np.array(["D2", "D2"]),
                    ),
                ],
            },
            sampling=sampling,
        )

        output = instance(event=input_data, add_event=add_event)
        print("@output", output, flush=True)

        # expected_output = NumpyEvent(
        #     data={
        #         (): [
        #             NumpyFeature(
        #                 name="sma_a",
        #                 data=np.array([10.0, 10.5, 11.0, 11.5, 14.0]),
        #             ),
        #             NumpyFeature(
        #                 name="sma_b",
        #                 data=np.array([20.0, 20.5, 21.0, 21.5, 24.0]),
        #             ),
        #         ]
        #     },
        #     sampling=NumpySampling(
        #         index=[],
        #         data={(): np.array([1, 2, 3, 5, 20], dtype=np.float64)},
        #     ),
        # )
        # self.assertEqual(repr(output), repr({"event": expected_output}))


if __name__ == "__main__":
    absltest.main()
