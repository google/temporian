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
from temporian.core.operators.multiply import MultiplyOperator
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling
from temporian.implementation.numpy.operators.multiply import (
    MultiplyNumpyImplementation,
)
from temporian.core.operators.sum import Resolution


class MultiplyOperatorTest(absltest.TestCase):
    def test_correct_multiplication(self) -> None:
        numpy_event_1_sampling = NumpySampling(
            names=["store_id"],
            data={("A",): np.array([1, 2, 3])},
        )

        numpy_event_1 = NumpyEvent(
            data={
                ("A",): [
                    NumpyFeature(
                        name="sales",
                        data=np.array([10.0, 11.0, 12.0]),
                    ),
                ],
            },
            sampling=numpy_event_1_sampling,
        )

        numpy_event_2 = NumpyEvent(
            data={
                ("A",): [
                    NumpyFeature(
                        name="costs",
                        data=np.array([1.0, 2.0, 3.0]),
                    ),
                ],
            },
            sampling=numpy_event_1_sampling,
        )

        numpy_output_sampling = NumpySampling(
            names=["store_id"],
            data={("A",): np.array([1, 2, 3])},
        )

        numpy_output_event = NumpyEvent(
            data={
                ("A",): [
                    NumpyFeature(
                        name="mult_sales_costs",
                        data=np.array([10.0, 22.0, 36.0]),
                    ),
                ],
            },
            sampling=numpy_output_sampling,
        )

        sampling = Sampling(["store_id"])
        event_1 = Event(
            [Feature("sales", float)],
            sampling=sampling,
            creator=None,
        )

        event_2 = Event(
            [Feature("costs", float)],
            sampling=sampling,
            creator=None,
        )

        operator = MultiplyOperator(
            event_1, event_2, Resolution.PER_FEATURE_IDX
        )
        mult_implementation = MultiplyNumpyImplementation(operator)
        output = mult_implementation(numpy_event_1, numpy_event_2)

        self.assertEqual(
            True,
            numpy_output_event == output["event"],
        )


if __name__ == "__main__":
    absltest.main()
