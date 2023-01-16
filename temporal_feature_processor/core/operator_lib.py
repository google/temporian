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

_OPERATORS = {}


def register_operator(operator_class):
  """Registers an operator."""

  definition = operator_class.build_op_definition()
  if definition.key in _OPERATORS:
    raise ValueError("Operator already registered")
  _OPERATORS[definition.key] = operator_class
