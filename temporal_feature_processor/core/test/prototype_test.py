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

import pandas as pd
from absl import logging
from absl.testing import absltest

from temporal_feature_processor.core import processor
from temporal_feature_processor.core.data.event import Event
from temporal_feature_processor.core.operators.assign import assign
from temporal_feature_processor.implementation.pandas.data.event import PandasEvent


class PrototypeTest(absltest.TestCase):

  assignee_pandas_event = PandasEvent({
      "product_id": [
          666964,
          666964,
          574016,
      ],
      "timestamp": [
          pd.Timestamp("2013-01-02", tz="UTC"),
          pd.Timestamp("2013-01-03", tz="UTC"),
          pd.Timestamp("2013-01-04",
                       tz="UTC"),  # identical timestamps for each index value
      ],
      "sales": [
          1091.0,
          919.0,
          953.0,
      ]
  }).set_index(["product_id", "timestamp"])

  assigned_pandas_event = PandasEvent({
      "product_id": [
          666964,
          666964,
          574016,
      ],
      "timestamp": [
          pd.Timestamp("2013-01-02", tz="UTC"),
          pd.Timestamp("2013-01-03", tz="UTC"),
          pd.Timestamp("2013-01-04",
                       tz="UTC"),  # identical timestamps for each index value
      ],
      "costs": [
          740.0,
          508.0,
          573.0,
      ]
  }).set_index(["product_id", "timestamp"])

  output_pandas_event = PandasEvent({
      "product_id": [
          666964,
          666964,
          574016,
      ],
      "timestamp": [
          pd.Timestamp("2013-01-02", tz="UTC"),
          pd.Timestamp("2013-01-03", tz="UTC"),
          pd.Timestamp("2013-01-04", tz="UTC"),
      ],
      "sales": [
          1091.0,
          919.0,
          953.0,
      ],
      "costs": [
          740.0,
          508.0,
          573.0,
      ]
  }).set_index(["product_id", "timestamp"])

  def test_prototoype(self) -> None:
    # get events
    assignee_event = self.assignee_pandas_event.schema()
    assigned_event = self.assigned_pandas_event.schema()
    expected_output_event = self.output_pandas_event.schema()

    # call assign operator
    output_event = assign(assignee_event, assigned_event)

    # pending features to compute. Intialize as output features
    pending_features = output_event.features().copy()

    # provide input materialized data
    assignee_event.set_data(self.assignee_pandas_event)
    assigned_event.set_data(self.assigned_pandas_event)

    # resolve graph
    while pending_features:
      logging.info(pending_features)
      feature = next(iter(pending_features))
      if feature.creator() is None:
        # is input feature
        pending_features.remove(feature)
        continue

      if feature.is_computed():
        # has already been computed
        pending_features.remove(feature)
        continue

      if all([
          creator_input_feature.is_computed()
          for creator_input_feature in feature.creator().inputs().values()
      ]):
        # evaluate operator
        feature.creator().evaluate()

      else:
        # some required features to evaluate the operator are missing.
        # Add them at the beggining of pending_features so they are visited on the
        # next while loop iteration
        pending_features = [
            creator_input_feature
            for creator_input_feature in feature.creator().inputs().values()
            if not creator_input_feature.is_computed()
        ] + pending_features

    self.assertEquals(True,
                      self.output_pandas_event.equals(output_event.data()))


if __name__ == "__main__":
  absltest.main()
