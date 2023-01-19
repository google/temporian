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

from temporal_feature_processor.core.data.event import Event
from temporal_feature_processor.core.data.feature import Feature


class PandasEvent(pd.DataFrame):

  @property
  def _constructor(self):
    return PandasEvent

  def schema(self) -> Event:
    features = [
        Feature(column_name, dtype=column_dtype)
        for column_name, column_dtype in self.dtypes.items()
    ]
    sampling = self.index.names
    return Event(features=features, sampling=sampling)
