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

"""PandasAssignOperator - different index test.

Tests that the assignee and assigned events cannot have different indexes.
"""

import pandas as pd

INPUT_1 = pd.DataFrame({
    "user_id": [
        151591562,
        193285921,
    ],
    "timestamp": [
        pd.Timestamp("2020-11-09"),
        pd.Timestamp("2020-11-10"),
    ],
    "price": [
        63.49,
        55.12,
    ],
}).set_index(["user_id", "timestamp"])  # index is user_id

INPUT_2 = pd.DataFrame({
    "product_id": [
        666964,
        574016,
    ],
    "timestamp": [
        pd.Timestamp("2020-11-09"),
        pd.Timestamp("2020-11-10"),
    ],
    "price": [
        126.98,
        266.42,
    ],
}).set_index(["product_id", "timestamp"])  # index is product_id
