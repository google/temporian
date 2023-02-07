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

from typing import List, Optional

from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling


class Event(object):
    def __init__(
        self,
        features: List[Feature],
        sampling: Sampling,
        name: Optional[str] = None,
    ):
        self._features = features
        self._sampling = sampling
        self._name = name

    def __repr__(self):
        return f"Event<features:{self._features},sampling:{self._sampling},id:{id(self)},name:{self._name}>"

    def sampling(self):
        return self._sampling

    def features(self):
        return self._features

    def name(self) -> str:
        return self._name

    def set_name(self, name) -> None:
        self._name = name


def input_event(
    features: List[Feature], index: List[str] = [], name: Optional[str] = None
) -> Event:

    sampling = Sampling(index=index, creator=None)

    for feature in features:
        if feature.sampling() is not None:
            raise ValueError(
                "Cannot call input_event on already linked features."
            )
        feature.set_sampling(sampling)

    return Event(
        features=features,
        sampling=sampling,
        name=name,
    )
