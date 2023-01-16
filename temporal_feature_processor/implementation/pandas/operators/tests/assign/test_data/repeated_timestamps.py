import pandas as pd

INPUT_1 = pd.DataFrame({
    "user_id": [
        151591562,
        151591562,
        151591562,
        191562515,
        191562515,
    ],
    "timestamp": [
        pd.Timestamp("2020-12-20", tz="UTC"),
        pd.Timestamp("2020-12-20", tz="UTC"),
        pd.Timestamp("2020-12-20", tz="UTC"),
        pd.Timestamp("2020-12-21", tz="UTC"),
        pd.Timestamp("2020-12-22", tz="UTC")
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
    "user_id": [
        151591562,
        151591562,
        151591562,
        191562515,
        191562515,
    ],
    "timestamp": [
        pd.Timestamp("2020-11-15", tz="UTC"),
        pd.Timestamp("2020-11-16", tz="UTC"),
        pd.Timestamp("2020-11-17", tz="UTC"),
        pd.Timestamp("2020-11-18", tz="UTC"),
        pd.Timestamp("2020-11-18", tz="UTC")
    ],
    "price": [
        190.47,
        190.47,
        190.47,
        399.63,
        446.01,
    ]
}).set_index([
    "user_id",
    "timestamp",
])
