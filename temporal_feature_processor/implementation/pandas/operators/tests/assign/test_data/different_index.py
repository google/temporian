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
        pd.Timestamp("2020-11-09", tz="UTC"),
        pd.Timestamp("2020-11-10", tz="UTC"),
    ],
    "price": [
        63.49,
        55.12,
    ]
}).set_index(["user_id", "timestamp"])  # index is user_id

INPUT_2 = pd.DataFrame({
    "product_id": [
        666964,
        574016,
    ],
    "timestamp": [
        pd.Timestamp("2020-11-09", tz="UTC"),
        pd.Timestamp("2020-11-10", tz="UTC"),
    ],
    "price": [
        126.98,
        266.42,
    ]
}).set_index(["product_id", "timestamp"])  # index is product_id
