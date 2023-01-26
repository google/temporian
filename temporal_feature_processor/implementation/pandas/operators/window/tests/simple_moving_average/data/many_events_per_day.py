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

"""Test data for SimpleMovingAverage operator."""

import pandas as pd

from temporal_feature_processor.implementation.pandas.data.event import PandasEvent
from temporal_feature_processor.implementation.pandas.data.sampling import PandasSampling

# several events for days 01 and 02
INPUT = PandasEvent(
    [
        [pd.Timestamp("2013-01-01 16:03:14"), 10.0],
        [pd.Timestamp("2013-01-01 16:04:42"), 20.0],
        [pd.Timestamp("2013-01-01 16:07:29"), 30.0],
        [pd.Timestamp("2013-01-02 16:10:41"), 40.0],
        [pd.Timestamp("2013-01-02 16:13:21"), 50.0],
        [pd.Timestamp("2013-01-03 16:10:41"), 60.0],
    ],
    columns=["timestamp", "value"],
).set_index(["timestamp"])

SAMPLING = PandasSampling.from_arrays(
    [[
        pd.Timestamp("2013-01-01"),
        pd.Timestamp("2013-01-02"),
        pd.Timestamp("2013-01-03"),
        # range goes up until day 04 since we have events during day 03
        # that wouldn't get processed otherwise
        pd.Timestamp("2013-01-04"),
    ]],
    names=["timestamp"])

OUTPUT = PandasEvent(
    [
        # first is empty since the output sampling's timestamps are in each day's start
        [pd.Timestamp("2013-01-01"), None],
        [pd.Timestamp("2013-01-02"), 20.0],
        [pd.Timestamp("2013-01-03"), 30.0],
        [pd.Timestamp("2013-01-04"), 35.0],
    ],
    columns=["timestamp", "sma_value"],
).set_index(["timestamp"])
