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
from temporian.core.operators.scalar import (
    equal_scalar,
    not_equal_scalar,
    greater_scalar,
    greater_equal_scalar,
    less_scalar,
    less_equal_scalar,
    add_scalar,
    subtract_scalar,
    multiply_scalar,
    divide_scalar,
    floordiv_scalar,
    modulo_scalar,
    power_scalar,
)


class ScalarTest(parameterized.TestCase):
    def setUp(self):
        self.evset = event_set(
            timestamps=[1, 2, 3, 4, 5],
            features={
                "index": [0, 0, 1, 1, 1],
                "f1": [10.0, 0.0, 12.0, np.nan, 30.0],
                "f2": ["A", "A", "B", "B", "C"],
            },
            indexes=["index"],
        )

    @parameterized.parameters(
        ("f1", equal_scalar, 12.0),
        ("f2", equal_scalar, "A"),
        ("f1", not_equal_scalar, 12.0),
        ("f1", greater_scalar, 11.0),
        ("f1", greater_equal_scalar, 11.0),
        ("f1", less_scalar, 11.0),
        ("f1", less_equal_scalar, 11.0),
    )
    def test_relational_generic(self, feature_name, operator, value):
        evtset = self.evset[feature_name]
        output_node = operator(evtset.node(), value)
        check_beam_implementation(
            self, input_data=evtset, output_node=output_node
        )

    @parameterized.parameters(
        add_scalar,
        subtract_scalar,
        multiply_scalar,
        divide_scalar,
        floordiv_scalar,
        modulo_scalar,
        power_scalar,
    )
    def test_arithmetic_generic(self, operator):
        evtset = self.evset["f1"]
        output_node = operator(evtset.node(), 2.0)
        check_beam_implementation(
            self, input_data=evtset, output_node=output_node
        )


if __name__ == "__main__":
    absltest.main()
