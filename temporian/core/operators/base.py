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

"""Base operator class and auxiliary classes definition."""

from abc import ABC
from typing import Any, Union

from temporian.core.data.event import Event
from temporian.proto import core_pb2 as pb


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
            exc_val.args += (f'In operator "{self._operator}".',)
        return False


class Operator(ABC):
    """Interface definition and common logic for operators."""

    def __init__(self):
        self._inputs: dict[str, Event] = {}
        self._outputs: dict[str, Event] = {}
        self._attributes: dict[str, Union[str, int]] = {}

    def __str__(self):
        return (
            f"Operator<key: {self.definition().key}, id: {id(self)},"
            f" attributes: {self.attributes()}>"
        )

    def outputs(self) -> dict[str, Event]:
        return self._outputs

    def inputs(self) -> dict[str, Event]:
        return self._inputs

    def attributes(self) -> dict[str, Any]:
        return self._attributes

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

    def add_attribute(self, key: str, value: Any) -> None:
        with OperatorExceptionDecorator(self):
            if key in self._attributes:
                raise ValueError(f'Already existing attribute "{key}".')
            self._attributes[key] = value

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        raise NotImplementedError()

    def definition(self):
        return self.__class__.build_op_definition()

    @classmethod
    def operator_key(cls):
        return cls.build_op_definition().key

    def set_inputs(self, inputs: dict[str, Event]) -> None:
        self._inputs = inputs

    def set_outputs(self, outputs: dict[str, Event]) -> None:
        self._outputs = outputs

    def set_attributes(self, attributes: dict[str, Event]) -> None:
        self._attributes = attributes

    def list_matching_io_samplings(self) -> list[tuple[str, str]]:
        """List pairs of input/output pairs with the same sampling.

        This function is used to check the correct implementation of ops are
        runtime.
        """
        # TODO: Optimize the number of matches: We don't need all the currently
        # computed matches to check the output validity.
        matches: list[tuple[str, str]] = []
        for output_key, output_value in self._outputs.items():
            for input_key, input_value in self._inputs.items():
                if output_value.sampling() is input_value.sampling():
                    matches.append((input_key, output_key))

        return matches
