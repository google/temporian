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


from absl.testing import absltest, parameterized

from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult


class DropTest(parameterized.TestCase):
    def test_drop_str(self):
        evset = event_set(
            timestamps=[1, 2, 3],
            features={
                "a": [1.0, 2.0, 3.0],
                "b": [5, 6, 7],
                "c": ["A", "A", "B"],
            },
            indexes=["c"],
        )
        result = evset.drop("a")

        expected = event_set(
            timestamps=[1, 2, 3],
            features={
                "b": [5, 6, 7],
                "c": ["A", "A", "B"],
            },
            indexes=["c"],
            same_sampling_as=evset,
        )

        assertOperatorResult(self, result, expected)

    def test_drop_list(self):
        evset = event_set(
            timestamps=[1, 2, 3],
            features={
                "a": [1.0, 2.0, 3.0],
                "b": [5, 6, 7],
                "c": ["A", "A", "B"],
            },
            indexes=["c"],
        )
        result = evset.drop(["a", "b"])

        expected = event_set(
            timestamps=[1, 2, 3],
            features={
                "c": ["A", "A", "B"],
            },
            indexes=["c"],
            same_sampling_as=evset,
        )

        assertOperatorResult(self, result, expected)


if __name__ == "__main__":
    absltest.main()
