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
from absl.testing.parameterized import TestCase

from temporian.core.compilation import compile
from temporian.core.serialization import save
from temporian.implementation.numpy.data.io import event_set
from temporian.test.utils import assertOperatorResult


class MapTest(TestCase):
    def test_basic(self):
        evset = event_set(timestamps=[1, 2, 3], features={"x": [10, 20, 30]})

        expected = event_set(
            timestamps=[1, 2, 3],
            features={"x": [20, 40, 60]},
            same_sampling_as=evset,
        )

        result = evset.map(lambda x: x * 2)

        assertOperatorResult(self, result, expected)

    def test_serialize_fails(self):
        @compile
        def f(e):
            return {"output": e.map(lambda x: x * 2)}

        evset = event_set([])

        with self.assertRaisesRegex(
            ValueError,
            (
                "Cannot serialize MAP operator since it takes a Python function"
                " as attribute."
            ),
        ):
            save(f, "path", evset)


if __name__ == "__main__":
    absltest.main()
