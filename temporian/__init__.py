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
# name, like this:
#
# from temporian.my_module import _private_name

# Core
from temporian.core import serialization
from temporian.core.data import dtype
from temporian.core.data import node
from temporian.core.data import feature
from temporian.core.data import sampling
from temporian.core.data import duration
from temporian.core.evaluation import evaluate
from temporian.core.operators import base
from temporian.core.operators.all_operators import *

# Implementation
from temporian.implementation.numpy.data.event_set import EventSet
from temporian.implementation.numpy.data.plotter import plot

# IO
from temporian.io.read_event_set import read_event_set
from temporian.io.save_event_set import save_event_set

# Operators registration mechanism
from temporian.core.operator_lib import registered_operators as _ops
from temporian.implementation.numpy.operators import all_operators as _impls

# Dtypes
float32 = dtype.DType.FLOAT32
float64 = dtype.DType.FLOAT64
int32 = dtype.DType.INT32
int64 = dtype.DType.INT64
bool_ = dtype.DType.BOOLEAN
str_ = dtype.DType.STRING

# Aliases
Feature = feature.Feature
load = serialization.load
save = serialization.save
input_node = node.input_node

__version__ = "0.0.1"
