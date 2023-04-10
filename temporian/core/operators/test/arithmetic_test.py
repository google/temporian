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

from temporian.core.data import event as event_lib
from temporian.core.data import dtype
from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators.arithmetic import (
    AdditionOperator,
    DivisionOperator,
    MultiplicationOperator,
    SubtractionOperator,
)


class ArithmeticOperatorsTest(absltest.TestCase):
    def setUp(self):
        self.sampling = Sampling(index=["x"])
        self.event_1 = event_lib.input_event(
            features=[
                Feature("f1", dtype.FLOAT32),
                Feature("f2", dtype.INT32),
            ],
            sampling=self.sampling,
        )
        self.event_2 = event_lib.input_event(
            features=[
                Feature("f3", dtype.FLOAT32),
                Feature("f4", dtype.INT32),
            ],
            sampling=self.sampling,
        )

    def test_addition(self):
        event_out = self.event_1 + self.event_2
        print(
            f"Creator: type={type(event_out.creator())} {event_out.creator()}"
        )
        assert isinstance(event_out.creator(), AdditionOperator)
        assert event_out.sampling() is self.sampling
        assert event_out.features()[0].creator() is event_out.creator()
        assert event_out.features()[1].creator() is event_out.creator()
        assert event_out.features()[0].name() == "add_f1_f3"
        assert event_out.features()[1].name() == "add_f2_f4"
        assert event_out.features()[0].dtype() == dtype.FLOAT32
        assert event_out.features()[1].dtype() == dtype.INT32

    def test_subtraction(self):
        event_out = self.event_1 - self.event_2
        assert isinstance(event_out.creator(), SubtractionOperator)
        assert event_out.sampling() is self.sampling
        assert event_out.features()[0].creator() is event_out.creator()
        assert event_out.features()[1].creator() is event_out.creator()
        assert event_out.features()[0].name() == "sub_f1_f3"
        assert event_out.features()[1].name() == "sub_f2_f4"
        assert event_out.features()[0].dtype() == dtype.FLOAT32
        assert event_out.features()[1].dtype() == dtype.INT32

    def test_multiplication(self):
        event_out = self.event_1 * self.event_2
        assert isinstance(event_out.creator(), MultiplicationOperator)
        assert event_out.sampling() is self.sampling
        assert event_out.features()[0].creator() is event_out.creator()
        assert event_out.features()[1].creator() is event_out.creator()
        assert event_out.features()[0].name() == "mult_f1_f3"
        assert event_out.features()[1].name() == "mult_f2_f4"
        assert event_out.features()[0].dtype() == dtype.FLOAT32
        assert event_out.features()[1].dtype() == dtype.INT32

    def test_division(self):
        event_out = self.event_1 / self.event_2
        assert isinstance(event_out.creator(), DivisionOperator)
        assert event_out.sampling() is self.sampling
        assert event_out.features()[0].creator() is event_out.creator()
        assert event_out.features()[1].creator() is event_out.creator()
        assert event_out.features()[0].name() == "div_f1_f3"
        assert event_out.features()[1].name() == "div_f2_f4"
        assert event_out.features()[0].dtype() == dtype.FLOAT64
        assert event_out.features()[1].dtype() == dtype.FLOAT64


if __name__ == "__main__":
    absltest.main()
