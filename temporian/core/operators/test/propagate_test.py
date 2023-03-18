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

import numpy as np

from temporian.core.data.sampling import Sampling
from temporian.core.operators.propagate import propagate
from temporian.core.data import event as event_lib
from temporian.core.data.feature import Feature
from temporian.core.data import dtype as dtype_lib


class PropagateOperatorTest(absltest.TestCase):
    def setUp(self):
        pass

    def test_basic(self):
        sampling = Sampling(index=["x"])
        event = event_lib.input_event(
            [
                Feature("a", dtype_lib.FLOAT64),
                Feature("b", dtype_lib.FLOAT64),
            ],
            sampling=sampling,
        )
        add_index = event_lib.input_event(
            [
                Feature("c", dtype_lib.STRING),
                Feature("d", dtype_lib.STRING),
            ],
            sampling=sampling,
        )
        _ = propagate(event=event, add_index=add_index)

    def test_str_add_event(self):
        sampling = Sampling(index=["x"])
        event = event_lib.input_event(
            [
                Feature("a", dtype_lib.FLOAT64),
                Feature("b", dtype_lib.FLOAT64),
                Feature("c", dtype_lib.STRING),
                Feature("d", dtype_lib.STRING),
            ],
            sampling=sampling,
        )
        _ = propagate(event=event, add_index=["c", "d"])

    def test_error_unknown_add_index(self):
        sampling = Sampling(index=["x"])
        event = event_lib.input_event(
            [
                Feature("a", dtype_lib.FLOAT64),
                Feature("b", dtype_lib.FLOAT64),
                Feature("c1", dtype_lib.STRING),
            ],
            sampling=sampling,
        )
        with self.assertRaisesRegex(KeyError, "{'c2'}"):
            _ = propagate(event=event, add_index=["c2"])

    def test_error_empty_add_index(self):
        sampling = Sampling(index=["x"])
        event = event_lib.input_event(
            [
                Feature("a", dtype_lib.FLOAT64),
                Feature("b", dtype_lib.FLOAT64),
                Feature("c1", dtype_lib.STRING),
            ],
            sampling=sampling,
        )
        with self.assertRaisesRegex(
            ValueError, "add_index contains no features"
        ):
            _ = propagate(event=event, add_index=[])

    def test_error_non_matching_sampling(self):
        event = event_lib.input_event(
            [
                Feature("a", dtype_lib.FLOAT64),
                Feature("b", dtype_lib.FLOAT64),
            ],
            sampling=Sampling(index=["x"]),
        )
        add_index = event_lib.input_event(
            [
                Feature("c", dtype_lib.STRING),
                Feature("d", dtype_lib.STRING),
            ],
            sampling=Sampling(index=["x"]),
        )
        with self.assertRaisesRegex(
            ValueError, "event and add_index should have the same sampling"
        ):
            _ = propagate(event=event, add_index=add_index)


if __name__ == "__main__":
    absltest.main()
