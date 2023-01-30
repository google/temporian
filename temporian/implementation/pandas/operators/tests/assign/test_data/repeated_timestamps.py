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

"""PandasAssignOperator - repeated timestamps test.

Tests that there cannot exist repeated timestamps on the assigned event for any
index value. Repeated timestamps are allowed on the assignee event.
"""

import pandas as pd

INPUT_1 = pd.DataFrame({
    "user_id": [
        151591562,
        151591562,
        191562515,
    ],
    "timestamp": [
        pd.Timestamp("2020-12-20"),
        pd.Timestamp(
            "2020-12-20"
        ),  # repeated timestamp on assingee event for user_id = 151591562
        pd.Timestamp("2020-12-22"),
    ],
    "price": [
        63.49,
        133.21,
        148.67,
    ],
}).set_index(["user_id", "timestamp"])

INPUT_2 = pd.DataFrame({
    "user_id": [
        151591562,
        191562515,
        191562515,
    ],
    "timestamp": [
        pd.Timestamp("2020-11-15"),
        pd.Timestamp("2020-11-18"),
        pd.Timestamp(
            "2020-11-18"
        ),  # repeated timestamp on assinged event for user_id = 191562515
    ],
    "price": [
        190.47,
        399.63,
        446.01,
    ],
}).set_index(["user_id", "timestamp"])
