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

"""A feature."""

from typing import Optional

from temporal_feature_processor.core.data import sampling as sampling_lib


class Feature(object):

  def __init__(self, name: str, dtype, sampling=None):
    self._name = name
    self._sampling = sampling
    self._dtype = dtype

  def name(self):
    return self._name

  def dtype(self):
    return self._dtype

  def sampling(self) -> Optional[sampling_lib.Sampling]:
    return self._sampling

  def set_sampling(self, sampling: sampling_lib.Sampling):
    self._sampling = sampling
