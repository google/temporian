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

from absl import logging
from absl.testing import absltest

import pandas as pd

import temporal_feature_processor as t


class TFPTest(absltest.TestCase):

  def test_does_nothing(self):
    logging.info("Running a test")
    t.does_nothing()
    self.assertEqual(1, 1)

  def test_create_toy_processor(self):
    p = t.core.create_toy_processor()
    logging.info("Processor:\n%s", p)

  def test_create_processor(self):
    a = t.place_holder(
        features=[
            t.Feature(name="f1", dtype=t.dtype.FLOAT),
            t.Feature(name="f2", dtype=t.dtype.FLOAT),
        ],
        index=[],
    )

    b = t.sma(data=a, window_length=7)

    input_signal_data = pd.DataFrame({
        "time": [0, 2, 4, 6],
        "f1": [1, 2, 3, 4],
        "f2": [5, 6, 7, 8],
    })

    results = t.Eval(
        query={"b": b},
        data={
            a: input_signal_data,
        },
    )

    logging.info("results: %s", results)


if __name__ == "__main__":
  absltest.main()
