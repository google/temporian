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


from absl.testing.parameterized import parameters
from temporian.core.operators.window.moving_sum import moving_sum
from temporian.core.operators.window.simple_moving_average import (
    simple_moving_average,
)
from absl.testing import absltest
from temporian.implementation.numpy.data.io import event_set
from temporian.beam.test.utils import check_beam_implementation


@parameters(moving_sum, simple_moving_average)
class BeamMovingSumTest(absltest.TestCase):
    def test_base(self, operator):
        # Create input data
        input_data = event_set(
            timestamps=[1, 2, 3, 4, 5, 1, 2, 3, 4],
            features={
                "a": ["x", "x", "x", "x", "x", "y", "y", "y", "y"],
                "b": [1, 1, 1, 2, 2, 1, 1, 1, 1],
                "c": [2.0, 3.0, 4.0, 3.0, 2.0, 22.0, 23.0, 24.0, 23.0],
                "d": [10.0, 11.0, 12.0, 13.0, 14.0, 105.0, 106.0, 106.0, 107.0],
                "e": [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0],
            },
            indexes=["a", "b"],
        )

        # Define computation
        output_node = operator(input_data.node(), 3)

        check_beam_implementation(
            self,
            input_data=input_data,
            output_node=output_node,
        )

    def test_with_sampling(self, operator):
        # Create input data
        input_data = event_set(
            timestamps=[1, 2, 3, 4, 5, 1, 2, 3, 4],
            features={
                "a": ["x", "x", "x", "x", "x", "y", "y", "y", "y"],
                "b": [1, 1, 1, 2, 2, 1, 1, 1, 1],
                "c": [2.0, 3.0, 4.0, 3.0, 2.0, 22.0, 23.0, 24.0, 23.0],
                "d": [10.0, 11.0, 12.0, 13.0, 14.0, 105.0, 106.0, 106.0, 107.0],
                "e": [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0],
            },
            indexes=["a", "b"],
        )

        sampling_data = event_set(
            timestamps=[-1, 1.5, 3.5, 1.5, 1.5, 1.5, 5],
            features={
                "a": ["x", "x", "x", "y", "y", "y", "z"],
                "b": [1, 1, 1, 1, 1, 1, 2],
            },
            indexes=["a", "b"],
        )

        # Define computation
        output_node = operator(
            input_data.node(), 3, sampling=sampling_data.node()
        )

        check_beam_implementation(
            self,
            input_data=[input_data, sampling_data],
            output_node=output_node,
        )


if __name__ == "__main__":
    absltest.main()
