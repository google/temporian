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
    ]
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
    ]
}).set_index(["product_id", "timestamp"])

OUTPUT = pd.DataFrame({
    "product_id": [
        666964,
        372306,
    ],
    "timestamp": [
        pd.Timestamp("2013-01-01", tz="UTC"),
        pd.Timestamp("2013-01-05",
                     tz="UTC"),  # assignee event's timestamps are preserved
    ],
    "sales": [
        0.0,
        1160.0,
    ],
    "costs": [
        0.0,
        508.0,
    ]
}).set_index(["product_id", "timestamp"])
