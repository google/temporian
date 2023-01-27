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

# two ids with the same exact timestamps
INPUT = PandasEvent(
    [
        [1, pd.Timestamp("2013-01-01"), 10.0],
        [1, pd.Timestamp("2013-01-02"), 20.0],
        [1, pd.Timestamp("2013-01-03"), 30.0],
        [1, pd.Timestamp("2013-01-12"), 40.0],
        [2, pd.Timestamp("2013-01-01"), 100.0],
        [2, pd.Timestamp("2013-01-02"), 200.0],
        [2, pd.Timestamp("2013-01-03"), 300.0],
        [2, pd.Timestamp("2013-01-12"), 400.0],
    ],
    columns=["id", "timestamp", "value"],
).set_index(["id", "timestamp"])

# sampling covering daily range of dates
SAMPLING = PandasSampling.from_tuples(
    [
        (1, pd.Timestamp("2013-01-01")),
        (1, pd.Timestamp("2013-01-02")),
        (1, pd.Timestamp("2013-01-03")),
        (1, pd.Timestamp("2013-01-04")),
        (1, pd.Timestamp("2013-01-05")),
        (1, pd.Timestamp("2013-01-06")),
        (1, pd.Timestamp("2013-01-07")),
        (1, pd.Timestamp("2013-01-08")),
        (1, pd.Timestamp("2013-01-09")),
        (1, pd.Timestamp("2013-01-10")),
        (1, pd.Timestamp("2013-01-11")),
        (1, pd.Timestamp("2013-01-12")),
        (2, pd.Timestamp("2013-01-01")),
        (2, pd.Timestamp("2013-01-02")),
        (2, pd.Timestamp("2013-01-03")),
        (2, pd.Timestamp("2013-01-04")),
        (2, pd.Timestamp("2013-01-05")),
        (2, pd.Timestamp("2013-01-06")),
        (2, pd.Timestamp("2013-01-07")),
        (2, pd.Timestamp("2013-01-08")),
        (2, pd.Timestamp("2013-01-09")),
        (2, pd.Timestamp("2013-01-10")),
        (2, pd.Timestamp("2013-01-11")),
        (2, pd.Timestamp("2013-01-12")),
    ],
    names=["id", "timestamp"],
)

OUTPUT = PandasEvent(
    [
        [1, pd.Timestamp("2013-01-01"), 10.0],
        [1, pd.Timestamp("2013-01-02"), 15.0],
        [1, pd.Timestamp("2013-01-03"), 20.0],
        [1, pd.Timestamp("2013-01-04"), 20.0],
        [1, pd.Timestamp("2013-01-05"), 20.0],
        [1, pd.Timestamp("2013-01-06"), 20.0],
        [1, pd.Timestamp("2013-01-07"), 20.0],
        [1, pd.Timestamp("2013-01-08"), 20.0],
        [1, pd.Timestamp("2013-01-09"), 25.0],
        [1, pd.Timestamp("2013-01-10"), 30.0],
        [1, pd.Timestamp("2013-01-11"), None],
        [1, pd.Timestamp("2013-01-12"), 40.0],
        [2, pd.Timestamp("2013-01-01"), 100.0],
        [2, pd.Timestamp("2013-01-02"), 150.0],
        [2, pd.Timestamp("2013-01-03"), 200.0],
        [2, pd.Timestamp("2013-01-04"), 200.0],
        [2, pd.Timestamp("2013-01-05"), 200.0],
        [2, pd.Timestamp("2013-01-06"), 200.0],
        [2, pd.Timestamp("2013-01-07"), 200.0],
        [2, pd.Timestamp("2013-01-08"), 200.0],
        [2, pd.Timestamp("2013-01-09"), 250.0],
        [2, pd.Timestamp("2013-01-10"), 300.0],
        [2, pd.Timestamp("2013-01-11"), None],
        [2, pd.Timestamp("2013-01-12"), 400.0],
    ],
    columns=["id", "timestamp", "sma_value"],
).set_index(["id", "timestamp"])
