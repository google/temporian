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

import temporian as tp
from temporian.implementation.numpy.data.event_set import EventSet


class CompileTest(absltest.TestCase):
    def test_basic(self):
        @tp.compile
        def f(x: EventSet):
            return tp.prefix("a_", x)

        evset = tp.event_set(
            timestamps=[1.0, 2.0, 3.0],
            features={"costs": [100.0, 200.0, 300.0]},
        )

        result = f(evset)

        self.assertEqual(type(result), EventSet)
        self.assertEqual(result.schema.feature_names(), ["a_costs"])

    def test_composed(self):
        @tp.compile
        def f(x: EventSet):
            return tp.glue(
                tp.prefix("a_", x),
                tp.prefix("b_", x),
            )

        evset = tp.event_set(
            timestamps=[1.0, 2.0, 3.0],
            features={"costs": [100.0, 200.0, 300.0]},
        )

        result = f(evset)

        self.assertEqual(type(result), EventSet)
        self.assertEqual(result.schema.feature_names(), ["a_costs", "b_costs"])


if __name__ == "__main__":
    absltest.main()
