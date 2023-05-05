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

from temporian.core import evaluation
from temporian.core import operator_lib
from temporian.core import processor
from temporian.core import serialize
from temporian.core.data import dtype
from temporian.core.data import node
from temporian.core.data import feature
from temporian.core.data import sampling
from temporian.core.data import duration
from temporian.core.operators import base
from temporian.io.read_event_set import read_event_set
from temporian.io.save_event_set import save_event_set

from temporian.implementation.numpy.data.event_set import EventSet

from temporian.implementation.numpy.data.plotter import plot

# Operators
from temporian.core.operators.all_operators import *

from temporian.core.operator_lib import registered_operators as get_operators

# Load all the implementations
from temporian.implementation.numpy.operators import all_operators as _impls

__version__ = "0.0.1"

evaluate = evaluation.evaluate
Feature = feature.Feature
load = serialize.load
save = serialize.save
input_node = node.input_node
Event = node.Node
