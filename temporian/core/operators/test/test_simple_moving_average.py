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
from temporian.core.operators.test.utils import _f32, _f64


class SimpleMovingAverageTest(absltest.TestCase):
    def test_empty(self):
        evset_f64 = event_set([], features={"a": _f64([])})
        evset_f32 = event_set([], features={"a": _f32([])})
        # a_f64 = _f64([])
        self.assertEqual(
            evset_f64.simple_moving_average(window_length=5.0),
            evset_f64,
        )
        self.assertEqual(
            evset_f32.simple_moving_average(window_length=5.0),
            evset_f32,
        )
        self.assertEqual(
            evset_f64.simple_moving_average(
                sampling=evset_f64, window_length=5.0
            ),
            evset_f64,
        )
        self.assertEqual(
            evset_f32.simple_moving_average(
                sampling=evset_f64, window_length=5.0
            ),
            evset_f32,
        )
