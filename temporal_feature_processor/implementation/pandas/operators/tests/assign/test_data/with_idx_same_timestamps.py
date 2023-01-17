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
        pd.Timestamp("2013-01-02", tz="UTC"),
        pd.Timestamp("2013-01-03", tz="UTC"),
        pd.Timestamp("2013-01-04",
                     tz="UTC"),  # identical timestamps for each index value
    ],
    "sales": [
        1091.0,
        919.0,
        953.0,
    ]
}).set_index(["product_id", "timestamp"])

INPUT_2 = pd.DataFrame({
    "product_id": [
        666964,
        666964,
        574016,
    ],
    "timestamp": [
        pd.Timestamp("2013-01-02", tz="UTC"),
        pd.Timestamp("2013-01-03", tz="UTC"),
        pd.Timestamp("2013-01-04",
                     tz="UTC"),  # identical timestamps for each index value
    ],
    "costs": [
        740.0,
        508.0,
        573.0,
    ]
}).set_index(["product_id", "timestamp"])

OUTPUT = pd.DataFrame({
    "product_id": [
        666964,
        666964,
        574016,
    ],
    "timestamp": [
        pd.Timestamp("2013-01-02", tz="UTC"),
        pd.Timestamp("2013-01-03", tz="UTC"),
        pd.Timestamp("2013-01-04", tz="UTC"),
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
    ]
}).set_index(["product_id", "timestamp"])
