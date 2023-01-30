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

from abc import ABC
from typing import Dict

from temporal_feature_processor.core.data.event import Event
from temporal_feature_processor.proto import core_pb2 as pb


class OperatorExceptionDecorator(object):
  """Adds details about an operator to exceptions raised in a block.

  Usage example:
    with OperatorExceptionDecorator(operator):
      raise ValueError("Something is wrong")

    Will print
      Something is wrong
      In operator NAME_OF_THE_OPERATOR
  """

  def __init__(self, operator):
    self._operator = operator

  def __enter__(self):
    pass

  def __exit__(self, exc_type, exc_val, traceback):
    if not exc_type:
      # No exceptions
      return True

    if exc_val:
      # Add operator details in the exception.
      exc_val.args += (
          (
              'In operator'
              f' "{self._operator.__class__.build_op_definition().key}".'
          ),
      )
    return False


class Operator(ABC):
  """Operator interface."""

  def __init__(self):
    self._inputs = {}
    self._outputs = {}

  def __str__(self):
    return f'Operator<key:{self.definition().key},id:{id(self)}>'

  def is_placeholder(self) -> bool:
    return self.definition().place_holder

  def outputs(self) -> Dict[str, Event]:
    return self._outputs

  def inputs(self) -> Dict[str, Event]:
    return self._inputs

  def check(self) -> None:
    """Ensures that the operator is valid."""

    definition = self.definition()

    with OperatorExceptionDecorator(self):
      # Check that expected inputs are present
      for expected_input in definition.inputs:
        if (
            not expected_input.is_optional
            and expected_input.key not in self._inputs
        ):
          raise ValueError(f'Missing input "{expected_input.key}".')

      # Check that no unexpected inputs are present
      for available_input in self._inputs:
        if available_input not in [v.key for v in definition.inputs]:
          raise ValueError(f'Unexpected input "{available_input}".')

      # Check that expected outputs are present
      for expected_output in definition.outputs:
        if expected_output.key not in self._outputs:
          raise ValueError(f'Missing output "{expected_output.key}".')

      # Check that no unexpected outputs are present
      for available_output in self._outputs:
        if available_output not in [v.key for v in definition.outputs]:
          raise ValueError(f'Unexpected output "{available_output}".')

  def add_input(self, key: str, event: Event) -> None:
    with OperatorExceptionDecorator(self):
      if key in self._inputs:
        raise ValueError(f'Already existing input "{key}".')
      self._inputs[key] = event

  def add_output(self, key: str, event: Event) -> None:
    with OperatorExceptionDecorator(self):
      if key in self._outputs:
        raise ValueError(f'Already existing output "{key}".')
      self._outputs[key] = event

  @classmethod
  def build_op_definition(cls) -> pb.OperatorDef:
    raise NotImplementedError()

  def definition(self):
    return self.__class__.build_op_definition()
