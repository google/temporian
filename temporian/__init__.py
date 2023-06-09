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

# pylint: disable=wrong-import-position
# pylint: disable=line-too-long

__version__ = "0.0.1"

# Register the ops definitions and implementations.
from temporian.implementation.numpy.operators import all_operators as _impls


# Actual API
# ==========

# Nodes and related
from temporian.core.data.node import Node
from temporian.core.data.node import input_node

# Dtypes
from temporian.core.data import dtype as _dtype

float32 = _dtype.DType.FLOAT32
float64 = _dtype.DType.FLOAT64
int32 = _dtype.DType.INT32
int64 = _dtype.DType.INT64
bool_ = _dtype.DType.BOOLEAN
str_ = _dtype.DType.STRING

# Schema
from temporian.core.data.schema import Schema

# Duration
# TODO: Only export the durations (e.g. milliseconds, seconds).
from temporian.core.data import duration


# Event set
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.data.io import event_set

# Graph serialization
from temporian.core.serialization import load
from temporian.core.serialization import save

# Graph execution
from temporian.core.evaluation import evaluate

# Operators
from temporian.core.operators.all_operators import *
from temporian.core.operator_lib import registered_operators as get_operators

# IO
from temporian.io.csv import to_csv
from temporian.io.csv import from_csv

from temporian.io.pandas import to_pandas
from temporian.io.pandas import from_pandas

# Plotting
from temporian.implementation.numpy.data.plotter import plot
