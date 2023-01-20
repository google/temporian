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

"""An event is a collection (possibly empty) of timesampled feature values."""

from typing import Any, List, Optional

from temporal_feature_processor.core.data.feature import Feature
from temporal_feature_processor.core.data.sampling import Sampling


class Event(object):

  def __init__(
      self,
      features: List[Feature],
      sampling: Sampling,
  ):
    self._features = features
    self._sampling = sampling

    # materialized data
    self._data: Optional[Any] = None

  def __repr__(self):
    return f'Event<features:{self._features},sampling:{self._sampling},id:{id(self)}>'

  def sampling(self):
    return self._sampling

  def features(self):
    return self._features

  def data(self) -> Optional[Any]:
    return self._data

  def set_data(self, data: Any) -> None:
    self._data = data

  def is_computed(self) -> bool:
    return self._data is not None
