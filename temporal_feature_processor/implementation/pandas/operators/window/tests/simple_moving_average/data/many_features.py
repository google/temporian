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

# two features with different magnitudes
INPUT = PandasEvent(
    [
        [pd.Timestamp("2013-01-01"), 10.0, 100.0],
        [pd.Timestamp("2013-01-02"), 20.0, 200.0],
        [pd.Timestamp("2013-01-03"), 30.0, 300.0],
        [pd.Timestamp("2013-01-12"), 40.0, 400.0],
    ],
    columns=["timestamp", "f1", "f2"],
).set_index(["timestamp"])

# sampling covering daily range of dates
SAMPLING = PandasSampling.from_arrays([[
    pd.Timestamp("2013-01-01"),
    pd.Timestamp("2013-01-02"),
    pd.Timestamp("2013-01-03"),
    pd.Timestamp("2013-01-04"),
    pd.Timestamp("2013-01-05"),
    pd.Timestamp("2013-01-06"),
    pd.Timestamp("2013-01-07"),
    pd.Timestamp("2013-01-08"),
    pd.Timestamp("2013-01-09"),
    pd.Timestamp("2013-01-10"),
    pd.Timestamp("2013-01-11"),
    pd.Timestamp("2013-01-12")
]],
                                      names=["timestamp"])

OUTPUT = PandasEvent(
    [
        [pd.Timestamp("2013-01-01"), 10.0, 100.0],
        [pd.Timestamp("2013-01-02"), 15.0, 150.0],
        [pd.Timestamp("2013-01-03"), 20.0, 200.0],
        [pd.Timestamp("2013-01-04"), 20.0, 200.0],
        [pd.Timestamp("2013-01-05"), 20.0, 200.0],
        [pd.Timestamp("2013-01-06"), 20.0, 200.0],
        [pd.Timestamp("2013-01-07"), 20.0, 200.0],
        [pd.Timestamp("2013-01-08"), 20.0, 200.0],
        [pd.Timestamp("2013-01-09"), 25.0, 250.0],
        [pd.Timestamp("2013-01-10"), 30.0, 300.0],
        [pd.Timestamp("2013-01-11"), None, None],
        [pd.Timestamp("2013-01-12"), 40.0, 400.0],
    ],
    columns=["timestamp", "sma_f1", "sma_f2"],
).set_index(["timestamp"])
