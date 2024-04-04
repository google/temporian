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

"""Registering mechanism for operator implementation classes."""

from typing import Any, Dict, Set

_OPERATOR_IMPLEMENTATIONS = {}

# TODO: Create a "registration" module to handle in-process and beam operator
# registration.


def check_operators_implementations_are_available(needed: Set[str]):
    """Checks if operator implementations are available."""
    missing = set(needed) - set(_OPERATOR_IMPLEMENTATIONS.keys())
    if missing:
        raise ValueError(
            f"Unknown operator implementations '{missing}' for Beam backend. It"
            " seems this operator is only available for the in-process"
            " Temporian backend. Available Beam operator implementations are:"
            f" {list(_OPERATOR_IMPLEMENTATIONS.keys())}."
        )


def register_operator_implementation(
    operator_class, operator_implementation_class
):
    """Registers an operator implementation."""
    op_key = operator_class.operator_key()
    if op_key in _OPERATOR_IMPLEMENTATIONS:
        raise ValueError("Operator implementation already registered")
    _OPERATOR_IMPLEMENTATIONS[op_key] = operator_implementation_class


def get_implementation_class(key: str):
    """Gets an operator implementation class from a registered key."""
    if key not in _OPERATOR_IMPLEMENTATIONS:
        raise ValueError(
            f"Unknown operator implementation '{key}' for Beam backend. It"
            " seems this operator is only available for the in-process"
            " Temporian backend. Available Beam operator implementations are:"
            f" {list(_OPERATOR_IMPLEMENTATIONS.keys())}."
        )
    return _OPERATOR_IMPLEMENTATIONS[key]


def registered_implementations() -> Dict[str, Any]:
    """Lists the registered operator implementations."""
    return _OPERATOR_IMPLEMENTATIONS
