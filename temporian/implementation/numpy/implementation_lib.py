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

"""Operator lib module."""

from typing import Any

_OPERATOR_IMPLEMENTATIONS = {}


def register_operator_implementation(
    operator_class, operator_implementation_class
):
    """Register an operator implementation."""

    op_key = operator_class.operator_key()
    if op_key in _OPERATOR_IMPLEMENTATIONS:
        raise ValueError("Operator implementation already registered")
    _OPERATOR_IMPLEMENTATIONS[op_key] = operator_implementation_class


def get_implementation_class(key: str):
    """Gets an operator implementation class from a registered key."""

    if key not in _OPERATOR_IMPLEMENTATIONS:
        raise ValueError(
            f"Unknown operator implementation '{key}'. Available operator "
            f"implementations are: {list(_OPERATOR_IMPLEMENTATIONS.keys())}."
        )
    return _OPERATOR_IMPLEMENTATIONS[key]


def registered_implementations() -> dict[str, Any]:
    """List the registered operator implementations."""

    return _OPERATOR_IMPLEMENTATIONS
