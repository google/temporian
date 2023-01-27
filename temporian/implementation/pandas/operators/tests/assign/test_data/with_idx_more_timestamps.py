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

Tests the correct output when the assigned event has more timestamps than the assignee 
event, for any index value. Both input events are indexed.
"""

import pandas as pd

INPUT_1 = pd.DataFrame({
    "product_id": [
        666964,
        372306,
    ],
    "timestamp": [
        pd.Timestamp("2013-01-01", tz="UTC"),
        pd.Timestamp("2013-01-05", tz="UTC"),
    ],
    "sales": [
        0.0,
        1160.0,
    ],
}).set_index(["product_id", "timestamp"])

INPUT_2 = pd.DataFrame({
    "product_id": [
        666964,
        372306,
        372306,
    ],
    "timestamp": [
        pd.Timestamp("2013-01-01", tz="UTC"),
        pd.Timestamp("2013-01-05", tz="UTC"),
        pd.Timestamp(
            "2013-01-07", tz="UTC"
        ),  # more timestamps than assignee event for porudct_id = 372306
    ],
    "costs": [
        0.0,
        508.0,
        573.0,
    ],
}).set_index(["product_id", "timestamp"])

OUTPUT = pd.DataFrame({
    "product_id": [
        666964,
        372306,
    ],
    "timestamp": [
        pd.Timestamp("2013-01-01", tz="UTC"),
        pd.Timestamp(
            "2013-01-05", tz="UTC"
        ),  # assignee event's timestamps are preserved
    ],
    "sales": [
        0.0,
        1160.0,
    ],
    "costs": [
        0.0,
        508.0,
    ],
}).set_index(["product_id", "timestamp"])
