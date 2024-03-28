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
from temporian.core.operators.filter_empty_index import filter_empty_index


class FilterEmptyIndexTest(absltest.TestCase):
    def test_basic(self):
        evset = event_set(
            timestamps=[1, 2, 3, 4],
            features={
                "i1": [1, 1, 2, 2],
                "f1": [10, 11, 12, 13],
            },
            indexes=["i1"],
        )

        input_data = evset.filter(evset["f1"] <= 11)
        output_node = filter_empty_index(input_data.node())

        check_beam_implementation(
            self, input_data=input_data, output_node=output_node
        )


if __name__ == "__main__":
    absltest.main()
