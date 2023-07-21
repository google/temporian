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

# type: ignore
# pylint: disable=wrong-import-position
# pylint: disable=line-too-long
# pylint: disable=no-name-in-module
# fmt: off

"""Temporian."""

# NOTE: If you need to import something here that isn't part of the public API,
# import it with a private name (and delete the symbol if possible):
# from temporian.module import submodule as _submodule
# del _submodule

__version__ = "0.1.2"

# Register all operator implementations
from temporian.implementation.numpy import operators as _impls
del _impls


# ================== #
# PUBLIC API SYMBOLS #
# ================== #

# EventSetNodes
from temporian.core.data.node import EventSetNode
from temporian.core.data.node import input_node

# Dtypes
from temporian.core.data.dtype import float64
from temporian.core.data.dtype import float32
from temporian.core.data.dtype import int32
from temporian.core.data.dtype import int64
from temporian.core.data.dtype import bool_
from temporian.core.data.dtype import str_
from temporian.core.data.dtype import bytes_

# Schema
from temporian.core.data.schema import Schema

# Durations
from temporian.core.data import duration

# EventSets
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.data.io import event_set

# Serialization
from temporian.core.serialization import save
from temporian.core.serialization import load
from temporian.core.serialization import save_graph
from temporian.core.serialization import load_graph

# Graph execution
from temporian.core.evaluation import run
from temporian.core.evaluation import has_leak

# IO
from temporian.io.csv import to_csv
from temporian.io.csv import from_csv
from temporian.io.pandas import to_pandas
from temporian.io.pandas import from_pandas
from temporian.io.tensorflow import to_tensorflow

# Plotting
from temporian.implementation.numpy.data.plotter import plot

# Compilation
from temporian.core.compilation import compile

# Types
from temporian.core.typing import EventSetOrNode

# --- OPERATORS ---

from temporian.core.operators.glue import glue
from temporian.core.operators.unique_timestamps import unique_timestamps
from temporian.core.operators.timestamps import timestamps

# Binary operators
from temporian.core.operators.binary.arithmetic import add
from temporian.core.operators.binary.arithmetic import subtract
from temporian.core.operators.binary.arithmetic import multiply
from temporian.core.operators.binary.arithmetic import divide
from temporian.core.operators.binary.arithmetic import floordiv
from temporian.core.operators.binary.arithmetic import modulo
from temporian.core.operators.binary.arithmetic import power

from temporian.core.operators.binary.relational import equal
from temporian.core.operators.binary.relational import not_equal
from temporian.core.operators.binary.relational import greater
from temporian.core.operators.binary.relational import greater_equal
from temporian.core.operators.binary.relational import less
from temporian.core.operators.binary.relational import less_equal

from temporian.core.operators.binary.logical import logical_and
from temporian.core.operators.binary.logical import logical_or
from temporian.core.operators.binary.logical import logical_xor

# Calendar operators
from temporian.core.operators.calendar.day_of_month import calendar_day_of_month
from temporian.core.operators.calendar.day_of_week import calendar_day_of_week
from temporian.core.operators.calendar.day_of_year import calendar_day_of_year
from temporian.core.operators.calendar.hour import calendar_hour
from temporian.core.operators.calendar.iso_week import calendar_iso_week
from temporian.core.operators.calendar.minute import calendar_minute
from temporian.core.operators.calendar.month import calendar_month
from temporian.core.operators.calendar.second import calendar_second
from temporian.core.operators.calendar.year import calendar_year

# Scalar operators
from temporian.core.operators.scalar.arithmetic_scalar import add_scalar
from temporian.core.operators.scalar.arithmetic_scalar import subtract_scalar
from temporian.core.operators.scalar.arithmetic_scalar import multiply_scalar
from temporian.core.operators.scalar.arithmetic_scalar import divide_scalar
from temporian.core.operators.scalar.arithmetic_scalar import floordiv_scalar
from temporian.core.operators.scalar.arithmetic_scalar import modulo_scalar
from temporian.core.operators.scalar.arithmetic_scalar import power_scalar

from temporian.core.operators.scalar.relational_scalar import equal_scalar
from temporian.core.operators.scalar.relational_scalar import not_equal_scalar
from temporian.core.operators.scalar.relational_scalar import greater_equal_scalar
from temporian.core.operators.scalar.relational_scalar import greater_scalar
from temporian.core.operators.scalar.relational_scalar import less_equal_scalar
from temporian.core.operators.scalar.relational_scalar import less_scalar

# Unary operators
from temporian.core.operators.unary import invert
from temporian.core.operators.unary import isnan
from temporian.core.operators.unary import notnan
from temporian.core.operators.unary import abs
from temporian.core.operators.unary import log

# Window operators
from temporian.core.operators.window.simple_moving_average import simple_moving_average
from temporian.core.operators.window.moving_standard_deviation import moving_standard_deviation
from temporian.core.operators.window.moving_sum import cumsum
from temporian.core.operators.window.moving_sum import moving_sum
from temporian.core.operators.window.moving_count import moving_count
from temporian.core.operators.window.moving_min import moving_min
from temporian.core.operators.window.moving_max import moving_max

# Remove automatic file tree symbols from public API
# pylint: disable=undefined-variable
del proto
del io
del core
del utils
del implementation
