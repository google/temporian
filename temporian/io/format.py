"""Format choices definitions and documentation for I/O functions."""

from enum import Enum
from typing import Literal, Union


# NOTE: DictEventSetFormatChoices' values in sync with enum (below)
class DictEventSetFormat(str, Enum):
    """Format choices for converting dictionaries to and from EventSets.

    The GROUPED_BY_INDEX value is generally recommended as it is more efficient
    than SINGLE_EVENTS.
    """

    GROUPED_BY_INDEX = "grouped_by_index"
    """Events in the same index are grouped together in a dictionary mapping
    index value, features and timestamps to actual values.

    In this dictionary, the features and timestamp keys are mapped to numpy
    arrays containing one value per event, and index keys are mapped to single
    value python primitives (e.g., `int`, `float`, `bytes`).

    The dtype of each numpy array matches the Temporian dtype. For instance, a
    Temporian feature with `dtype=tp.int32` is stored as a numpy array with
    `dtype=np.int32`.

    For example, an EventSet with 3 events and the following Schema:

    ```
    features=[("f1", tp.int64), ("f2", tp.str_)]
    indexes=[("i1", tp.int64), ("i2", tp.str_)]
    ```

    would be represented as the following dictionary:

    ```
    {
    "timestamp": np.array([100.0, 101.0, 102.0], np.float64),
    "f1": np.array([1, 2, 3], np.int64),
    "f2": np.array([b"a", b"b", b"c"], np.bytes_),
    "i1": 10,
    "i2": b"x",
    }
    ```
    """

    SINGLE_EVENTS = "single_events"
    """Each event is represented as an individual dictionary of keys to unique
    values. Each index value, feature and timestamp is represented by an
    independent dictionary.

    For example, the same EventSet with 3 events and the following Schema:

    ```
    features=[("f1", tp.int64), ("f2", tp.str_)]
    indexes=[("i1", tp.int64), ("i2", tp.str_)]
    ```

    would be represented as the following dictionaries:

    ```
    {"timestamp": 100.0, "f1": 1, "f2": b"a", "i1": 10, "i2": b"x"}
    {"timestamp": 101.0, "f1": 2, "f2": b"b", "i1": 10, "i2": b"x"}
    {"timestamp": 102.0, "f1": 3, "f2": b"c", "i1": 10, "i2": b"x"}
    ```
    """


DictEventSetFormatChoices = Union[
    DictEventSetFormat, Literal["grouped_by_index", "single_events"]
]
