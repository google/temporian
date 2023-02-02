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

import temporian
from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling


class Event(object):
    def __init__(
        self,
        features: List[Feature],
        sampling: Sampling,
        # TODO: make Operator the creator's type. I don't know how to circumvent
        # the cyclical import error
        creator: Optional[Any] = None,
    ):
        self._features = features
        self._sampling = sampling
        self._creator = creator

    def __getitem__(self, feature_names: List[str]) -> "Event":
        # instance select operator. Import from base temporian package
        # to avoid circular import
        select_operator = temporian.core.operators.select.SelectOperator(
            self, feature_names
        )
        # return Event
        return select_operator.outputs()["event"]

    def __repr__(self) -> str:
        features_print = "\n\t\t".join(
            [str(feature) for feature in self._features]
        )
        return (
            "Event: { \n"
            "\tfeatures: {\n"
            f"\t\t{features_print}\n"
            "\t},\n"
            f"\tsampling: {self._sampling},\n"
            f"\tid:{id(self)}\n}}"
        )

    def sampling(self):
        return self._sampling

    def features(self):
        return self._features

    def creator(self):
        return self._creator
