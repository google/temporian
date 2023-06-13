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

"""Temporian."""

# NOTE: The API reference documentation reads this file and expects a single
# import per line. Do not import several symbols from the same module in a
# single line. Do not allow import to break lines.

# NOTE: If you need to import something here that isn't part of the public API,
# and therefore shouldn't show up in the documentation, import it with a private
# name:
# from temporian.module import submodule as _submodule

# NOTE: Wildcard imports (*) are treated and parsed as part of root-level
# imports, so same rules apply to modules imported with a wildcard.
# Also:
# - Modules imported with wildcards are shown as subfolders in the docs.
# - Wildcard imports are only allowed for modules, which must have an init file.

# pylint: disable=wrong-import-position
# pylint: disable=line-too-long

__version__ = "0.0.1"

# Register the ops definitions and implementations.
from temporian.core.operator_lib import registered_operators as _ops
from temporian.implementation.numpy import operators as _impls


# Actual API
# ==========

# Nodes and related
from temporian.core.data.node import Node
from temporian.core.data.node import input_node

# Dtypes
from temporian.core.data.dtypes.api_symbols import *

# Schema
from temporian.core.data.schema import Schema

# Durations
from temporian.core.data import duration

# Event set
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.data.io import event_set

# Graph serialization
from temporian.core.serialization.api_symbols import *

# Graph execution
from temporian.core.evaluation import evaluate

# Operators
from temporian.core.operators.api_symbols import *

# IO
from temporian.io.api_symbols import *

# Plotting
from temporian.implementation.numpy.data.plotter import plot
