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
from temporian.core.operators.rename import rename


class RenameTest(absltest.TestCase):
    def setUp(self):
        self.evset = event_set(
            timestamps=[1, 2],
            features={"a": [10, 11]},
        )

    def test_rename_single_feature_with_str(self):
        output_node = rename(self.evset.node(), "b")
        check_beam_implementation(
            self, input_data=self.evset, output_node=output_node
        )

    def test_rename_single_feature_with_dict(self):
        output_node = rename(self.evset.node(), {"a": "b"})
        check_beam_implementation(
            self, input_data=self.evset, output_node=output_node
        )

    def test_rename_multiple_features(self):
        evset = event_set(
            timestamps=[1, 2],
            features={"a": [10, 11], "b": [1, 2], "c": [100, 101]},
        )
        output_node = rename(evset.node(), {"a": "d", "b": "e"})
        check_beam_implementation(
            self, input_data=evset, output_node=output_node
        )

    def test_rename_multiple_features_with_list(self):
        evset = event_set(
            timestamps=[1, 2],
            features={"a": [10, 11], "b": [1, 2], "c": [100, 101]},
        )
        output_node = rename(evset.node(), ["d", "e", "f"])
        check_beam_implementation(
            self, input_data=evset, output_node=output_node
        )

    def test_rename_single_index_with_str(self):
        evset = event_set(
            timestamps=[1, 2],
            features={"a": [10, 11], "b": [1, 2]},
            indexes=["b"],
        )
        output_node = rename(evset.node(), indexes="c")
        check_beam_implementation(
            self, input_data=evset, output_node=output_node
        )

    def test_rename_single_index_with_dict(self):
        evset = event_set(
            timestamps=[1, 2],
            features={"a": [10, 11], "b": [1, 2]},
            indexes=["b"],
        )
        output_node = rename(evset.node(), indexes={"b": "c"})
        check_beam_implementation(
            self, input_data=evset, output_node=output_node
        )

    def test_rename_multiple_indexes(self):
        evset = event_set(
            timestamps=[1, 2],
            features={"a": [10, 11], "b": [1, 2]},
            indexes=["a", "b"],
        )
        output_node = rename(evset.node(), indexes={"a": "c", "b": "d"})
        check_beam_implementation(
            self, input_data=evset, output_node=output_node
        )

    def test_rename_feature_and_index_inverting_name(self) -> None:
        evset = event_set(
            timestamps=[1, 2],
            features={"a": [10, 11], "b": [1, 2]},
            indexes=["b"],
        )
        output_node = rename(
            evset.node(),
            features={"a": "b"},
            indexes={"b": "a"},
        )
        check_beam_implementation(
            self, input_data=evset, output_node=output_node
        )


if __name__ == "__main__":
    absltest.main()
