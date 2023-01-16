import pandas as pd

INPUT_1 = pd.DataFrame({
    "product_id": [
        666964,
        666964,
        666964,
        574016,
        372306,
    ],
    "timestamp": [
        pd.Timestamp("2013-01-01", tz="UTC"),
        pd.Timestamp("2013-01-02", tz="UTC"),
        pd.Timestamp("2013-01-03", tz="UTC"),
        pd.Timestamp("2013-01-04", tz="UTC"),
        pd.Timestamp("2013-01-05", tz="UTC"),
    ],
    "sales": [
        0.0,
        1091.0,
        919.0,
        953.0,
        1160.0,
    ]
}).set_index([
    "product_id",
    "timestamp",
])

INPUT_2 = pd.DataFrame({
    "product_id": [
        666964,
        666964,
        666964,
        574016,
        372306,
    ],
    "timestamp": [
        pd.Timestamp("2013-01-01", tz="UTC"),
        pd.Timestamp("2013-01-02", tz="UTC"),
        pd.Timestamp("2013-01-03", tz="UTC"),
        pd.Timestamp("2013-01-04", tz="UTC"),
        pd.Timestamp("2013-01-05", tz="UTC"),
    ],
    "costs": [
        0.0,
        740.0,
        508.0,
        573.0,
        790.0,
    ]
}).set_index([
    "product_id",
    "timestamp",
])

OUTPUT = pd.DataFrame({
    "product_id": [
        666964,
        666964,
        666964,
        574016,
        372306,
    ],
    "timestamp": [
        pd.Timestamp("2013-01-01", tz="UTC"),
        pd.Timestamp("2013-01-02", tz="UTC"),
        pd.Timestamp("2013-01-03", tz="UTC"),
        pd.Timestamp("2013-01-04", tz="UTC"),
        pd.Timestamp("2013-01-05", tz="UTC"),
    ],
    "sales": [
        0.0,
        1091.0,
        919.0,
        953.0,
        1160.0,
    ],
    "costs": [
        0.0,
        740.0,
        508.0,
        573.0,
        790.0,
    ]
}).set_index([
    "product_id",
    "timestamp",
])
