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
from temporian.core.operators.propagate import Propagate
from temporian.implementation.numpy.operators.propagate import (
    PropagateNumpyImplementation,
)
from temporian.implementation.numpy.data.event import NumpyEvent, NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling
from temporian.core.data import event as event_lib
from temporian.core.data import feature as feature_lib
from temporian.core.data.dtype import DType


class PropagateOperatorTest(absltest.TestCase):
    def setUp(self):
        pass

    def test_base(self):
        # Define input
        sampling = Sampling(index_levels=[("x", DType.STRING)])
        event = event_lib.input_event(
            [
                feature_lib.Feature(name="a", dtype=DType.FLOAT64),
                feature_lib.Feature(name="b", dtype=DType.FLOAT64),
            ],
            sampling=sampling,
        )
        to = event_lib.input_event(
            [
                feature_lib.Feature(name="c", dtype=DType.INT64),
                feature_lib.Feature(name="d", dtype=DType.INT64),
            ],
            sampling=sampling,
        )

        # Create input data
        # TODO: Use "from_dataframe" when it suppose sampling sharing.

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
                    NumpyFeature("a", np.array([1, 2, 3], dtype=np.float64)),
                    NumpyFeature("b", np.array([4, 5, 6], dtype=np.float64)),
                ],
                ("X2",): [
                    NumpyFeature("a", np.array([7, 8], dtype=np.float64)),
                    NumpyFeature("b", np.array([9, 10], dtype=np.float64)),
                ],
            },
            sampling=sampling,
        )

        to_data = NumpyEvent(
            data={
                ("X1",): [
                    NumpyFeature("c", np.array([1, 2, 1], dtype=np.int64)),
                    NumpyFeature("d", np.array([1, 1, 2], dtype=np.int64)),
                ],
                ("X2",): [
                    NumpyFeature("c", np.array([1, 1], dtype=np.int64)),
                    NumpyFeature("d", np.array([2, 1], dtype=np.int64)),
                ],
            },
            sampling=sampling,
        )

        # Expected output
        expected_sampling = NumpySampling(
            index=["x", "c", "d"],
            data={
                ("X1", 1, 1): np.array([0.1, 0.2, 0.3], dtype=np.float64),
                ("X1", 1, 2): np.array([0.1, 0.2, 0.3], dtype=np.float64),
                ("X1", 2, 1): np.array([0.1, 0.2, 0.3], dtype=np.float64),
                ("X2", 1, 1): np.array([0.4, 0.5], dtype=np.float64),
                ("X2", 1, 2): np.array([0.4, 0.5], dtype=np.float64),
            },
        )

        expected_output = NumpyEvent(
            data={
                (
                    "X1",
                    1,
                    1,
                ): [
                    NumpyFeature("a", np.array([1, 2, 3], dtype=np.float64)),
                    NumpyFeature("b", np.array([4, 5, 6], dtype=np.float64)),
                ],
                (
                    "X1",
                    1,
                    2,
                ): [
                    NumpyFeature("a", np.array([1, 2, 3], dtype=np.float64)),
                    NumpyFeature("b", np.array([4, 5, 6], dtype=np.float64)),
                ],
                (
                    "X1",
                    2,
                    1,
                ): [
                    NumpyFeature("a", np.array([1, 2, 3], dtype=np.float64)),
                    NumpyFeature("b", np.array([4, 5, 6], dtype=np.float64)),
                ],
                ("X2", 1, 1): [
                    NumpyFeature("a", np.array([7, 8], dtype=np.float64)),
                    NumpyFeature("b", np.array([9, 10], dtype=np.float64)),
                ],
                ("X2", 1, 2): [
                    NumpyFeature("a", np.array([7, 8], dtype=np.float64)),
                    NumpyFeature("b", np.array([9, 10], dtype=np.float64)),
                ],
            },
            sampling=expected_sampling,
        )

        # Run op
        op = Propagate(event=event, to=to)
        instance = PropagateNumpyImplementation(op)
        output = instance.call(event=input_data, to=to_data)["event"]
        self.assertEqual(output, expected_output)


if __name__ == "__main__":
    absltest.main()
