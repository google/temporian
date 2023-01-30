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
import pandas as pd

from temporal_feature_processor.core import evaluator
from temporal_feature_processor.core.data.event import Event
from temporal_feature_processor.core.data.event import Feature
from temporal_feature_processor.core.data.sampling import Sampling
from temporal_feature_processor.core.operators.assign import assign
from temporal_feature_processor.implementation.pandas.data import event as pandas_event


class PrototypeTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()

    self.assignee_event = "tensorflow_decision_forests/contrib/temporal_feature_processor/temporal_feature_processor/test/test_data/prototype/assignee_event.csv"

    self.assigned_event = pandas_event.PandasEvent({
        "product_id": [666964, 666964, 574016],
        "timestamp": [
            pd.Timestamp("2013-01-02", tz="UTC"),
            pd.Timestamp("2013-01-03", tz="UTC"),
            pd.Timestamp("2013-01-04", tz="UTC"),
        ],
        "costs": [740.0, 508.0, 573.0],
    }).set_index(["product_id", "timestamp"])

    self.expected_output_event = pandas_event.PandasEvent({
        "product_id": [666964, 666964, 574016],
        "timestamp": [
            pd.Timestamp("2013-01-02", tz="UTC"),
            pd.Timestamp("2013-01-03", tz="UTC"),
            pd.Timestamp("2013-01-04", tz="UTC"),
        ],
        "sales": [1091.0, 919.0, 953.0],
        "costs": [740.0, 508.0, 573.0],
    }).set_index(["product_id", "timestamp"])

  def test_prototoype(self) -> None:
    # instance input events
    assignee_event = Event(
        features=[Feature(name="sales", dtype=float)],
        sampling=Sampling(["product_id", "timestamp"]),
    )
    assigned_event = Event(
        features=[Feature(name="costs", dtype=float)],
        sampling=Sampling(["product_id", "timestamp"]),
    )

    # call assign operator
    output_event = assign(assignee_event, assigned_event)

    # evaluate output
    output_event_pandas = evaluator.evaluate(
        output_event,
        input_data={
            # assignee event specified from disk
            assignee_event: self.assignee_event,
            # assigned event loaded in ram
            assigned_event: self.assigned_event,
        },
        backend="pandas",
    )

    print(self.assignee_event)
    print(self.assigned_event)
    print(self.expected_output_event)
    print(output_event_pandas)

    # validate
    self.assertEqual(
        True,
        self.expected_output_event.equals(output_event_pandas[output_event]),
    )


if __name__ == "__main__":
  absltest.main()
