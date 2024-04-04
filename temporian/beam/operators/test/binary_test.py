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

import numpy as np
from absl.testing import absltest, parameterized

from temporian.implementation.numpy.data.io import event_set
from temporian.beam.test.utils import check_beam_implementation
from temporian.core.operators.binary import (
    add,
    subtract,
    multiply,
    divide,
    floordiv,
    modulo,
    power,
    equal,
    not_equal,
    greater,
    greater_equal,
    less,
    less_equal,
    logical_and,
    logical_or,
    logical_xor,
)


class BinaryTest(parameterized.TestCase):
    def setUp(self):
        self.evset = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={
                "index": [0, 0, 1, 1, 1],
                "n1": [10.0, 0.0, 12.0, np.nan, 4.0],
                "n2": [11.0, 0.0, 11.0, np.nan, 5.0],
                "s1": ["A", "A", "B", "B", "C"],
                "s2": ["A", "B", "B", "C", "C"],
                "b1": [True, True, False, False, False],
                "b2": [True, False, True, False, False],
            },
            indexes=["index"],
        )

    @parameterized.parameters(
        add,
        subtract,
        multiply,
        divide,
        floordiv,
        modulo,
        power,
        equal,
        not_equal,
        greater,
        greater_equal,
        less,
        less_equal,
    )
    def test_binary_numerical_generic(self, operator):
        e1 = self.evset["n1"]
        e2 = self.evset["n2"]
        output_node = operator(e1.node(), e2.node())
        check_beam_implementation(
            self, input_data=[e1, e2], output_node=output_node
        )

    @parameterized.parameters(
        logical_and,
        logical_or,
        logical_xor,
    )
    def test_binary_boolan_generic(
        self,
        operator,
    ):
        e1 = self.evset["b1"]
        e2 = self.evset["b2"]
        output_node = operator(e1.node(), e2.node())
        check_beam_implementation(
            self, input_data=[e1, e2], output_node=output_node
        )

    @parameterized.parameters(
        equal,
        not_equal,
    )
    def test_binary_string_generic(self, operator):
        e1 = self.evset["s1"]
        e2 = self.evset["s2"]
        output_node = operator(e1.node(), e2.node())
        check_beam_implementation(
            self, input_data=[e1, e2], output_node=output_node
        )


if __name__ == "__main__":
    absltest.main()
