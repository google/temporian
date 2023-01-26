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

from temporal_feature_processor.core.data.event import Event
from temporal_feature_processor.core.data.event import Feature
from temporal_feature_processor.core.data.sampling import Sampling
from temporal_feature_processor.core.operators.assign import assign
from temporal_feature_processor.core import evaluator
from temporal_feature_processor.implementation.pandas.data import event as pandas_event


class PrototypeTest(absltest.TestCase):

  def setUp(self) -> None:

    self.expected_output_event = pandas_event.pandas_event_from_csv(
        "temporal_feature_processor/test/test_data/prototype/output_event.csv",
        Sampling(["product_id", "timestamp"]))

  def test_prototoype(self) -> None:

    # instance input events
    assignee_event = Event(features=[Feature(name="sales", dtype=float)],
                           sampling=Sampling(["product_id", "timestamp"]))
    assigned_event = Event(features=[Feature(name="costs", dtype=float)],
                           sampling=Sampling(["product_id", "timestamp"]))

    # call assign operator
    output_event = assign(assignee_event, assigned_event)

    # evaluate output
    output_event_pandas = evaluator.evaluate(
        output_event,
        data={
            assignee_event:
                "temporal_feature_processor/test/test_data/prototype/assignee_event.csv",
            assigned_event:
                "temporal_feature_processor/test/test_data/prototype/assigned_event.csv",
        },
        backend="pandas")

    # validate
    self.assertEqual(
        True,
        self.expected_output_event.equals(output_event_pandas[output_event]))


if __name__ == "__main__":
  absltest.main()
