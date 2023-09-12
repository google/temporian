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

"""Imports all the packages containing operator implementations."""

# pylint: disable=unused-import
# pylint: disable=line-too-long
# fmt: off
from temporian.implementation.numpy.operators import cast
from temporian.implementation.numpy.operators import combine
from temporian.implementation.numpy.operators import drop_index
from temporian.implementation.numpy.operators import filter
from temporian.implementation.numpy.operators import glue
from temporian.implementation.numpy.operators import join
from temporian.implementation.numpy.operators import unary
from temporian.implementation.numpy.operators import lag
from temporian.implementation.numpy.operators import leak
from temporian.implementation.numpy.operators import prefix
from temporian.implementation.numpy.operators import propagate
from temporian.implementation.numpy.operators import select
from temporian.implementation.numpy.operators import add_index
from temporian.implementation.numpy.operators import resample
from temporian.implementation.numpy.operators import rename
from temporian.implementation.numpy.operators.binary import arithmetic
from temporian.implementation.numpy.operators.binary import relational
from temporian.implementation.numpy.operators.binary import logical

from temporian.implementation.numpy.operators.scalar import arithmetic_scalar
from temporian.implementation.numpy.operators.scalar import relational_scalar

from temporian.implementation.numpy.operators.window import simple_moving_average
from temporian.implementation.numpy.operators.window import moving_standard_deviation
from temporian.implementation.numpy.operators.window import moving_sum
from temporian.implementation.numpy.operators.window import moving_count
from temporian.implementation.numpy.operators.window import moving_min
from temporian.implementation.numpy.operators.window import moving_max
from temporian.implementation.numpy.operators.calendar import day_of_month
from temporian.implementation.numpy.operators.calendar import day_of_week
from temporian.implementation.numpy.operators.calendar import day_of_year
from temporian.implementation.numpy.operators.calendar import year
from temporian.implementation.numpy.operators.calendar import month
from temporian.implementation.numpy.operators.calendar import iso_week
from temporian.implementation.numpy.operators.calendar import hour
from temporian.implementation.numpy.operators.calendar import minute
from temporian.implementation.numpy.operators.calendar import second

from temporian.implementation.numpy.operators import begin
from temporian.implementation.numpy.operators import end
from temporian.implementation.numpy.operators import enumerate
from temporian.implementation.numpy.operators import fast_fourier_transform
from temporian.implementation.numpy.operators import select_index_values
from temporian.implementation.numpy.operators import since_last
from temporian.implementation.numpy.operators import tick
from temporian.implementation.numpy.operators import timestamps
from temporian.implementation.numpy.operators import unique_timestamps
from temporian.implementation.numpy.operators import filter_max_moving_count
