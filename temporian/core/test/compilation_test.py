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

from typing import Dict, List, Tuple
from absl.testing import absltest

import temporian as tp
from temporian.implementation.numpy.data.event_set import EventSet


class CompileTest(absltest.TestCase):
    def setUp(self):
        self.evset = tp.event_set(
            timestamps=[1.0, 2.0, 3.0],
            features={"a": [100.0, 200.0, 300.0]},
        )
        self.other_evset = tp.event_set(
            timestamps=[1.0, 2.0, 3.0],
            features={"b": [100.0, 200.0, 300.0]},
        )

    def test_basic(self):
        @tp.compile
        def f(x: EventSet) -> EventSet:
            return tp.prefix("a", x)

        result = f(self.evset)

        self.assertEqual(type(result), EventSet)
        self.assertEqual(result.schema.feature_names(), ["aa"])

    def test_composed(self):
        @tp.compile
        def f(x: EventSet) -> EventSet:
            return tp.glue(
                tp.prefix("a", x),
                tp.prefix("b", x),
            )

        result = f(self.evset)

        self.assertEqual(type(result), EventSet)
        self.assertEqual(result.schema.feature_names(), ["aa", "ba"])

    def test_other_args(self):
        @tp.compile
        def f(a: int, x: EventSet, b: str) -> EventSet:
            print(a, b)
            return tp.prefix("a", x)

        result = f(1, self.evset, "a")

        self.assertEqual(type(result), EventSet)
        self.assertEqual(result.schema.feature_names(), ["aa"])

    def test_tuple_arg(self):
        @tp.compile
        def f(x: Tuple[EventSet]) -> EventSet:
            return tp.prefix("a", x[0])

        result = f((self.evset, self.other_evset))

        self.assertEqual(type(result), EventSet)
        self.assertEqual(result.schema.feature_names(), ["aa"])

    def test_list_arg(self):
        @tp.compile
        def f(x: List[EventSet]) -> EventSet:
            return tp.prefix("a", x[0])

        result = f([self.evset, self.other_evset])

        self.assertEqual(type(result), EventSet)
        self.assertEqual(result.schema.feature_names(), ["aa"])

    def test_dict_arg(self):
        @tp.compile
        def f(x: Dict[str, EventSet]) -> EventSet:
            return tp.prefix("a", list(x.values())[0])

        result = f({"a": self.evset, "b": self.other_evset})

        self.assertEqual(type(result), EventSet)
        self.assertEqual(result.schema.feature_names(), ["aa"])


if __name__ == "__main__":
    absltest.main()
