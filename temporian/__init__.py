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

from temporian.core import core
from temporian.core import evaluator
from temporian.core import operator_lib
from temporian.core import operators
from temporian.core import processor
from temporian.core.data import dtype
from temporian.core.data import event
from temporian.core.data import feature
from temporian.core.data import sampling
from temporian.core.operators import base

__version__ = "0.0.1"

does_nothing = core.does_nothing
create_toy_processor = core.create_toy_processor

sma = operators.simple_moving_average.sma
place_holder = operators.place_holder.place_holder
evaluate = evaluator.evaluate
Feature = feature.Feature
