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

"""Evaluator module."""


from typing import List, Union, Dict
import pandas as pd


from temporal_feature_processor import event as event_lib

Query = Union[
    event_lib.Event, Dict[str, event_lib.Event], List[event_lib.Event]
]

Data = Union[pd.DataFrame, Dict[str, pd.DataFrame], List[pd.DataFrame]]

Result = Data


def Eval(
    query: Query,
    data: Data,
) -> Result:
  """Evaluates a query on data."""

  del query
  del data
