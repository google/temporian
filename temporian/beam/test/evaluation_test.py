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

from temporian.core.operators.window.moving_sum import moving_sum
from temporian.implementation.numpy.data.io import event_set
from temporian.beam.test.utils import check_beam_implementation
from temporian.core.operators.select import select


class IOTest(absltest.TestCase):
    def test_run(self):
        # Create input data
        input_data = event_set(
            timestamps=[1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            features={
                "a": ["x", "x", "x", "x", "x", "y", "y", "y", "y", "y"],
                "b": [1, 1, 1, 2, 2, 1, 1, 1, 1, 1],
                "c": [2, 3, 4, 3, 2, 22, 23, 24, 23, 22],
                "d": [100, 101, 102, 103, 104, 105, 106, 106, 107, 108],
                "e": [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
            },
            indexes=["a", "b"],
        )

        # Define computation
        output_node = select(moving_sum(input_data.node(), 3), "d")

        check_beam_implementation(
            self,
            input_data=input_data,
            output_node=output_node,
        )


if __name__ == "__main__":
    absltest.main()
