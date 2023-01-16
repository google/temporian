import pandas as pd

INPUT_1 = pd.DataFrame({
    "user_id": [
        151591562,
        151591562,
        151591562,
        151591562,
        151591562,
    ],
    "timestamp": [
        pd.Timestamp("2020-11-09", tz="UTC"),
        pd.Timestamp("2020-11-10", tz="UTC"),
        pd.Timestamp("2020-11-10", tz="UTC"),
        pd.Timestamp("2020-11-11", tz="UTC"),
        pd.Timestamp("2020-11-12", tz="UTC"),
    ],
    "price": [
        63.49,
        63.49,
        63.49,
        133.21,
        148.67,
    ]
}).set_index([
    "user_id",
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
        pd.Timestamp("2020-11-09", tz="UTC"),
        pd.Timestamp("2020-11-10", tz="UTC"),
        pd.Timestamp("2020-11-10", tz="UTC"),
        pd.Timestamp("2020-11-11", tz="UTC"),
        pd.Timestamp("2020-11-12", tz="UTC"),
    ],
    "price": [
        126.98,
        126.98,
        126.98,
        266.42,
        297.34,
    ]
}).set_index([
    "product_id",
    "timestamp",
])
