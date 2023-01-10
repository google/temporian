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

"""Temporal Feature Processor."""

from temporal_feature_processor import core
from temporal_feature_processor import dtype
from temporal_feature_processor import evaluator
from temporal_feature_processor import event
from temporal_feature_processor import feature
from temporal_feature_processor import operator
from temporal_feature_processor import operator_lib
from temporal_feature_processor import operators
from temporal_feature_processor import processor


__version__ = "0.0.1"

does_nothing = core.does_nothing

sma = operators.simple_moving_average.sma
place_holder = operators.place_holder.place_holder
Eval = evaluator.Eval
Feature = feature.Feature
