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


from typing import Dict
from abc import ABC, abstractmethod

from temporian.core.operators.base import Operator
from temporian.beam.io import PEventSet


class BeamOperatorImplementation(ABC):
    def __init__(self, operator: Operator):
        assert operator is not None
        self._operator = operator

    @property
    def operator(self):
        return self._operator

    @abstractmethod
    def call(self, **inputs: PEventSet) -> Dict[str, PEventSet]:
        pass

    def __call__(self, **inputs: PEventSet) -> Dict[str, PEventSet]:
        outputs = self.call(**inputs)
        return outputs
