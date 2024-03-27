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
from temporian.core.operators.drop_index import drop_index


class DropIndexTest(absltest.TestCase):
    def setUp(self) -> None:
        self.evset = event_set(
            timestamps=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            features={
                "a": ["A", "A", "A", "B", "B", "B", "B", "B"],
                "b": [0, 0, 0, 0, 0, 1, 1, 1],
                "c": [1, 1, 1, 2, 2, 2, 2, 3],
                "d": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            },
            indexes=["b", "c"],
        )

    def test_drop_all(self):
        output_node = drop_index(self.evset.node())
        check_beam_implementation(
            self, input_data=self.evset, output_node=output_node
        )

    def test_drop_single_second(self) -> None:
        output_node = drop_index(self.evset.node(), "c")
        check_beam_implementation(
            self, input_data=self.evset, output_node=output_node
        )

    def test_drop_all_dont_keep(self):
        output_node = drop_index(self.evset.node(), keep=False)
        check_beam_implementation(
            self, input_data=self.evset, output_node=output_node
        )

    def test_drop_single_second_dont_keep(self) -> None:
        output_node = drop_index(self.evset.node(), "c", keep=False)
        check_beam_implementation(
            self, input_data=self.evset, output_node=output_node
        )

    def test_drop_all_dont_keep_no_features(self):
        evset = event_set(
            timestamps=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            features={
                "b": [0, 0, 0, 0, 0, 1, 1, 1],
                "c": [1, 1, 1, 2, 2, 2, 2, 3],
            },
            indexes=["b", "c"],
        )
        output_node = drop_index(evset.node(), keep=False)
        check_beam_implementation(
            self, input_data=evset, output_node=output_node
        )

    def test_drop_all_no_features(self):
        evset = event_set(
            timestamps=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            features={
                "b": [0, 0, 0, 0, 0, 1, 1, 1],
                "c": [1, 1, 1, 2, 2, 2, 2, 3],
            },
            indexes=["b", "c"],
        )
        output_node = drop_index(evset.node())
        check_beam_implementation(
            self, input_data=evset, output_node=output_node
        )


if __name__ == "__main__":
    absltest.main()
