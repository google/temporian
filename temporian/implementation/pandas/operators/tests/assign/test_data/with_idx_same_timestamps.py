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

"""PandasAssignOperator - with index, more timestamps test.

Tests the correct output when the assignee and assigned events have identical
timestamps for all index values. Both input events are indexed.
"""

import pandas as pd

INPUT_1 = pd.DataFrame({
    "product_id": [
        666964,
        666964,
        574016,
    ],
    "timestamp": [
        pd.Timestamp("2013-01-02"),
        pd.Timestamp("2013-01-03"),
        pd.Timestamp("2013-01-04"),  # identical timestamps for each index value
    ],
    "sales": [
        1091.0,
        919.0,
        953.0,
    ],
}).set_index(["product_id", "timestamp"])

INPUT_2 = pd.DataFrame({
    "product_id": [
        666964,
        666964,
        574016,
    ],
    "timestamp": [
        pd.Timestamp("2013-01-02"),
        pd.Timestamp("2013-01-03"),
        pd.Timestamp("2013-01-04"),  # identical timestamps for each index value
    ],
    "costs": [
        740.0,
        508.0,
        573.0,
    ],
}).set_index(["product_id", "timestamp"])

OUTPUT = pd.DataFrame({
    "product_id": [
        666964,
        666964,
        574016,
    ],
    "timestamp": [
        pd.Timestamp("2013-01-02"),
        pd.Timestamp("2013-01-03"),
        pd.Timestamp("2013-01-04"),
    ],
    "sales": [
        1091.0,
        919.0,
        953.0,
    ],
    "costs": [
        740.0,
        508.0,
        573.0,
    ],
}).set_index(["product_id", "timestamp"])
