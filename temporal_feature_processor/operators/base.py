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

"""Operator module."""

from abc import ABC, abstractmethod
from typing import Dict

from temporal_feature_processor.data.event import Event
from temporal_feature_processor.implementation.pandas.operators.base import (
    PandasOperator,)


class Operator(ABC):
  """Operator interface."""

  def __init__(self):
    self._inputs = {}
    self._outputs = {}

  def outputs(self) -> Dict[str, Event]:
    return self._outputs

  def inputs(self) -> Dict[str, Event]:
    return self._inputs

  def check(self) -> None:
    """Ensures that the operator is valid."""
    pass

  def add_input(self, key: str, event: Event) -> None:
    if key in self._inputs:
      raise ValueError(f"Input {key} already existing")
    self._inputs[key] = event

  def add_output(self, key: str, event: Event) -> None:
    if key in self._outputs:
      raise ValueError(f"Output {key} already existing")
    self._outputs[key] = event

  @abstractmethod
  def _get_pandas_implementation(self) -> PandasOperator:
    """Get pandas implementation for this operator."""
