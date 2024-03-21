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
from temporian.test.utils import assertOperatorResult, f64, i32


class FilterEmptyIndexTest(parameterized.TestCase):
    def test_basic(self):
        evset = event_set(
            timestamps=[1, 2, 3, 4],
            features={
                "i1": [1, 1, 2, 2],
                "f1": [10, 11, 12, 13],
            },
            indexes=["i1"],
        )

        result = evset.filter(evset["f1"] <= 11).filter_empty_index()
        expected = event_set(
            timestamps=[1, 2],
            features={
                "i1": [1, 1],
                "f1": [10, 11],
            },
            indexes=["i1"],
        )
        assertOperatorResult(self, result, expected, check_sampling=False)


if __name__ == "__main__":
    absltest.main()
