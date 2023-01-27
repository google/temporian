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

from temporian.core.data.event import Event
from temporian.core.data.feature import Feature


class PandasFeature(pd.Series):

  @property
  def _constructor(self):
    return PandasFeature

  @property
  def _constructor_expanddim(self):
    return PandasEvent

  def schema(self) -> Feature:
    return Feature(name=self.name, dtype=self.dtype, sampling=self.index.names)


class PandasEvent(pd.DataFrame):

  @property
  def _constructor(self):
    return PandasEvent

  @property
  def _constructor_sliced(self):
    return PandasFeature

  def schema(self) -> Event:
    return Event(
        features=[self[col].schema() for col in self.columns],
        sampling=self.index.names,
    )
