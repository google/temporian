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
from typing import Union, Any

from temporian.core.data.node import Node
from temporian.proto import core_pb2 as pb


# Valid types for operator attributes
AttributeType = Union[str, int, float, bool, list[str], dict[str, str]]


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
        self._inputs: dict[str, Node] = {}
        self._outputs: dict[str, Node] = {}
        self._attributes: dict[str, AttributeType] = {}
        self._definition: pb.OperatorDef = self.build_op_definition()
        self._attr_types: dict[str, type] = {
            attr.key: attr.type for attr in self._definition.attributes
        }

    def __str__(self):
        return (
            f"Operator<key: {self.definition().key}, id: {id(self)},"
            f" attributes: {self.attributes}>"
        )

    @property
    def attributes(self) -> dict[str, AttributeType]:
        return self._attributes

    @property
    def inputs(self) -> dict[str, Node]:
        return self._inputs

    @property
    def outputs(self) -> dict[str, Node]:
        return self._outputs

    @attributes.setter
    def attributes(self, attributes: dict[str, AttributeType]):
        self._attributes = attributes

    @inputs.setter
    def inputs(self, inputs: dict[str, Node]):
        self._inputs = inputs

    @outputs.setter
    def outputs(self, outputs: dict[str, Node]):
        self._outputs = outputs

    def add_input(self, key: str, node: Node) -> None:
        with OperatorExceptionDecorator(self):
            if key in self.inputs:
                raise ValueError(f'Already existing input "{key}".')
            self.inputs[key] = node

    def add_output(self, key: str, node: Node) -> None:
        with OperatorExceptionDecorator(self):
            if key in self.outputs:
                raise ValueError(f'Already existing output "{key}".')
            self.outputs[key] = node

    def add_attribute(self, key: str, value: AttributeType) -> None:
        with OperatorExceptionDecorator(self):
            if key in self.attributes:
                raise ValueError(f'Already existing attribute "{key}".')
            self.attributes[key] = self.cast_attribute_type(
                value, self._attr_types[key]
            )

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

            # Check that expected attributes are present and have correct type
            for expected_attr in definition.attributes:
                # From definition
                def_key = expected_attr.key
                def_type = expected_attr.type

                if def_key not in self._attributes:
                    raise ValueError(f'Missing attr. "{expected_attr.key}".')

                # Check that the value type is as defined for this attribute
                attr_value = self._attributes[def_key]
                self.check_attribute_type(attr_value, def_type)

            # Check that no unexpected attributes are present
            for available_attr in self._attributes:
                if available_attr not in [v.key for v in definition.attributes]:
                    raise ValueError(f'Unexpected attr. "{available_attr}".')

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        raise NotImplementedError()

    def definition(self) -> pb.OperatorDef:
        return self._definition

    @classmethod
    def operator_key(cls) -> str:
        return cls.build_op_definition().key

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
                if output_value.sampling_node is input_value.sampling_node:
                    matches.append((input_key, output_key))

        return matches

    @classmethod
    def check_attribute_type(
        cls, value: AttributeType, attr_type: pb.OperatorDef.Attribute.Type
    ) -> None:
        """
        Check that the value given for an attribute matches the type
        that is defined for protobuf serialization.
        """

        # Helper functions
        def is_dict_str(value):
            return (
                isinstance(value, dict)
                and all(isinstance(v, str) for v in value.values())
                and all(isinstance(k, str) for k in value.keys())
            )

        def is_list_str(value):
            return isinstance(value, list) and all(
                isinstance(v, str) for v in value
            )

        # Check exact matching between attr type (except ANY) and value type
        if (
            attr_type == pb.OperatorDef.Attribute.Type.STRING
            and not isinstance(value, str)
        ):
            raise ValueError(f"Attribute {value=} mismatch: string expected")
        if (
            attr_type == pb.OperatorDef.Attribute.Type.INTEGER_64
            and not isinstance(value, int)
        ):
            raise ValueError(f"Attribute {value=} mismatch: integer expected")
        if (
            attr_type == pb.OperatorDef.Attribute.Type.FLOAT_64
            and not isinstance(value, float)
        ):
            raise ValueError(f"Attribute {value=} mismatch: float expected")
        if attr_type == pb.OperatorDef.Attribute.Type.BOOL and not isinstance(
            value, bool
        ):
            raise ValueError(f"Attribute {value=} mismatch: bool expected")
        if (
            attr_type == pb.OperatorDef.Attribute.Type.REPEATED_STRING
            and not is_list_str(value)
        ):
            raise ValueError(
                f"Attribute {value=} type mismatch: list[str] expected"
            )
        if (
            attr_type == pb.OperatorDef.Attribute.Type.MAP_STR_STR
            and not is_dict_str(value)
        ):
            raise ValueError(
                f"Attribute {value=} type mismatch: dict[str,str] expected"
            )

        # Special case: ANY attribute type, still needs to be a valid type
        if (
            attr_type == pb.OperatorDef.Attribute.Type.ANY
            and not isinstance(value, str)
            and not isinstance(value, bool)
            and not isinstance(value, int)
            and not isinstance(value, float)
            and not is_list_str(value)
            and not is_dict_str(value)
        ):
            raise ValueError(
                "Attribute of type ANY has an invalid value type:"
                f" {type(value)}"
            )

    @classmethod
    def cast_attribute_type(
        cls, value: Any, attr_type: pb.OperatorDef.Attribute.Type
    ) -> Any:
        """
        Cast some attribute types that can be converted without risk:
        int -> float
        int [0,1] -> bool
        """
        if attr_type == pb.OperatorDef.Attribute.Type.FLOAT_64 and isinstance(
            value, int
        ):
            return float(value)
        if (
            attr_type == pb.OperatorDef.Attribute.Type.BOOL
            and isinstance(value, int)
            and value in [0, 1]
        ):
            return bool(value)
        return value
