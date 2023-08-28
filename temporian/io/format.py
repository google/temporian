"""Format of EventSet when importing/exporting them."""


def help_grouped_by_index():
    """Prints the documentation of the `grouped_by_index` arg. used in IO."""

    print(
        """
`grouped_by_index` controls whether events in the same index key are stored together or as independent items in an IO container. The default value `grouped_by_index=True` is generally recommended as it is more efficient than `grouped_by_index=False`.

With grouped_by_index=True
==========================

Events in the same index are grouped together in a dictionary mapping features, index and timestamps to actual values. In this dictionary, features and the timestamps keys are mapped to numpy arrays containing one value per event. Indexe keys are mapped to single value python primitives (e.g., int, float, bytes). Type dtype of this numpy array matches the Temporian dtype. For instance, a Temporian feature with dtype=tp.int32 a stored as a numpy array with dtype=np.int32.

For example:

Schema
    features=[("f1", tp.int64), ("f2", tp.str_)]
    indexes=[("i1", tp.int64), ("i2", tp.str_)]

One dictionary containing three events.
    {
    "timestamp": np.array([100.0, 101.0, 102.0], np.float64),
    "f1": np.array([1, 2, 3], np.int64),
    "f2": np.array([b"a", b"b", b"c"], np.bytes_),
    "i1": 10,
    "i2": b"x",
    }


With grouped_by_index=False
===========================

Each event is represented as an individual dictionary of keys to unique values. Each feature, index as well as the timestamp are one entry in the dictionary.

For example:

Schema
features=[("f1", tp.int64), ("f2", tp.str_)]
indexes=[("i1", tp.int64), ("i2", tp.str_)]

Same three events represented as three dictionary items.

{"timestamp": 100.0, "f1": 1, "f2": b"a", "i1": 10, "i2": b"x"}
{"timestamp": 101.0, "f1": 2, "f2": b"b", "i1": 10, "i2": b"x"}
{"timestamp": 102.0, "f1": 3, "f2": b"c", "i1": 10, "i2": b"x"}
"""
    )
