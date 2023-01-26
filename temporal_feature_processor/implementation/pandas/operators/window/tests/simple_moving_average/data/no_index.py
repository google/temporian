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

from temporal_feature_processor.implementation.pandas.data.event import \
    PandasEvent
from temporal_feature_processor.implementation.pandas.data.sampling import \
    PandasSampling

INPUT = PandasEvent(
    [
        [pd.Timestamp("2013-01-01", tz="UTC"), 10.0],
        [pd.Timestamp("2013-01-02", tz="UTC"), 20.0],
        [pd.Timestamp("2013-01-03", tz="UTC"), 30.0],
        [pd.Timestamp("2013-01-12", tz="UTC"), 40.0],
    ],
    columns=["timestamp", "value"],
).set_index(["timestamp"])

SAMPLING = PandasSampling.from_arrays([[
    pd.Timestamp("2013-01-01", tz="UTC"),
    pd.Timestamp("2013-01-02", tz="UTC"),
    pd.Timestamp("2013-01-03", tz="UTC"),
    pd.Timestamp("2013-01-04", tz="UTC"),
    pd.Timestamp("2013-01-05", tz="UTC"),
    pd.Timestamp("2013-01-06", tz="UTC"),
    pd.Timestamp("2013-01-07", tz="UTC"),
    pd.Timestamp("2013-01-08", tz="UTC"),
    pd.Timestamp("2013-01-09", tz="UTC"),
    pd.Timestamp("2013-01-10", tz="UTC"),
    pd.Timestamp("2013-01-11", tz="UTC"),
    pd.Timestamp("2013-01-12", tz="UTC")
]],
                                      names=["timestamp"])

OUTPUT = PandasEvent(
    [
        [pd.Timestamp("2013-01-01", tz="UTC"), 10.0],
        [pd.Timestamp("2013-01-02", tz="UTC"), 15.0],
        [pd.Timestamp("2013-01-03", tz="UTC"), 20.0],
        [pd.Timestamp("2013-01-04", tz="UTC"), 20.0],
        [pd.Timestamp("2013-01-05", tz="UTC"), 20.0],
        [pd.Timestamp("2013-01-06", tz="UTC"), 20.0],
        [pd.Timestamp("2013-01-07", tz="UTC"), 20.0],
        [pd.Timestamp("2013-01-08", tz="UTC"), 20.0],
        [pd.Timestamp("2013-01-09", tz="UTC"), 25.0],
        [pd.Timestamp("2013-01-10", tz="UTC"), 30.0],
        [pd.Timestamp("2013-01-11", tz="UTC"), None],
        [pd.Timestamp("2013-01-12", tz="UTC"), 40.0],
    ],
    columns=["timestamp", "simple_moving_average"],
).set_index(["timestamp"])
