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

# WARNING: The API reference documentation reads this file and expects a single
# import per line. Do not import several symbols from the same module in a
# single line.

# TIP: If you need to import something here that isn't part of the public API,
# and therefore shouldn't show up in the documentation, import it with a private
# name:
# from temporian.module import submodule as _submodule

# pylint: disable=wrong-import-position

__version__ = "0.0.1"

# Register the ops definitions and implementations.
from temporian.implementation.numpy.operators import all_operators as _impls


# Actual API
# ==========

# Nodes and related
from temporian.core.data import node as _node

Node = _node.Node
input_node = _node.input_node

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
from temporian.core import serialization as _serialization

load = _serialization.load
save = _serialization.save

# Graph execution
from temporian.core.evaluation import evaluate

# Operators
from temporian.core.operators.all_operators import *
from temporian.core.operator_lib import registered_operators as get_operators

# IO
from temporian.io import csv as _csv

to_csv = _csv.to_csv
from_csv = _csv.from_csv

from temporian.io import pandas as _pandas

to_pandas = _pandas.to_pandas
from_pandas = _pandas.from_pandas

# Plotting
from temporian.implementation.numpy.data.plotter import plot
