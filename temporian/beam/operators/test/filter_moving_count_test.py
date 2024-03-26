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
from temporian.core.operators.filter_moving_count import filter_moving_count


class FilterMovingCountTest(absltest.TestCase):
    def test_without_feature(self):
        input_data = event_set([1, 2, 4], {})
        output_node = filter_moving_count(input_data.node(), 1.5)
        check_beam_implementation(
            self, input_data=input_data, output_node=output_node
        )

    def test_with_feature(self):
        input_data = event_set([1, 2, 4], {"f": [10, 11, 14]})
        output_node = filter_moving_count(input_data.node(), 1.5)
        check_beam_implementation(
            self, input_data=input_data, output_node=output_node
        )


if __name__ == "__main__":
    absltest.main()
