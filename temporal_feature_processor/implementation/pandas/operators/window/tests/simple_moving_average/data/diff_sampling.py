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
        [1, pd.Timestamp("2013-01-01"), 10.0],
        [1, pd.Timestamp("2013-01-02"), 20.0],
        [1, pd.Timestamp("2013-01-03"), 30.0],
        [1, pd.Timestamp("2013-01-12"), 40.0],
        [2, pd.Timestamp("2013-01-11"), 100.0],
        [2, pd.Timestamp("2013-01-12"), 200.0],
        [2, pd.Timestamp("2013-01-13"), 300.0],
        [2, pd.Timestamp("2013-01-14"), 400.0],
    ],
    columns=["id", "timestamp", "value"],
).set_index(["id", "timestamp"])

SAMPLING = PandasSampling.from_tuples([
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
    (2, pd.Timestamp("2013-01-11")),
    (2, pd.Timestamp("2013-01-12")),
    (2, pd.Timestamp("2013-01-13")),
    (2, pd.Timestamp("2013-01-14")),
],
                                      names=["id", "timestamp"])

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
        [2, pd.Timestamp("2013-01-11"), 100.0],
        [2, pd.Timestamp("2013-01-12"), 150.0],
        [2, pd.Timestamp("2013-01-13"), 200.0],
        [2, pd.Timestamp("2013-01-14"), 250.0],
    ],
    columns=["id", "timestamp", "simple_moving_average"],
).set_index(["id", "timestamp"])
