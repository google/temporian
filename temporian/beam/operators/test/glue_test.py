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

from temporian.implementation.numpy.data.io import event_set
from temporian.beam.test.utils import check_beam_implementation
from temporian.core.operators.glue import glue


class GlueTest(absltest.TestCase):
    def test_basic(self):
        timestamps = [1, 1, 2, 3, 4]

        evset_1 = event_set(
            timestamps=timestamps,
            features={
                "x": ["a", "a", "a", "a", "b"],
                "f1": [10, 11, 12, 13, 14],
            },
            indexes=["x"],
        )
        evset_2 = event_set(
            timestamps=timestamps,
            features={
                "x": ["a", "a", "a", "a", "b"],
                "f2": [20, 21, 22, 23, 24],
                "f3": [30, 31, 32, 33, 34],
            },
            indexes=["x"],
            same_sampling_as=evset_1,
        )
        evset_3 = event_set(
            timestamps=timestamps,
            features={
                "x": ["a", "a", "a", "a", "b"],
                "f4": [40, 41, 42, 43, 44],
            },
            indexes=["x"],
            same_sampling_as=evset_1,
        )

        output_node = glue(evset_1.node(), evset_2.node(), evset_3.node())

        check_beam_implementation(
            self,
            input_data=[evset_1, evset_2, evset_3],
            output_node=output_node,
        )


if __name__ == "__main__":
    absltest.main()
