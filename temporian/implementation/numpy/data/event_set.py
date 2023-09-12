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

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
)
import sys

import numpy as np
from temporian.implementation.numpy.data.dtype_normalization import (
    _DTYPE_REVERSE_MAPPING,
    normalize_index_key,
)

from temporian.utils import config
from temporian.core.data.node import (
    EventSetNode,
    create_node_with_new_reference,
)
from temporian.core.data.schema import Schema
from temporian.core.event_set_ops import EventSetOperations

if TYPE_CHECKING:
    from temporian.core.typing import IndexKey, NormalizedIndexKey
    from temporian.core.operators.base import Operator


@dataclass
class IndexData:
    """Features and timestamps data for a single index group.

    Note: The `schema` constructor argument is only used for checking. If
    `schema=None`, no checking is done. Checking can be done manually with
    `index_data.check_schema(...)`.

    Attributes:
        features: List of one-dimensional NumPy arrays representing the
            features.
        timestamps: One-dimensional NumPy array representing the timestamps.

    Usage example:
        ```
        >>> features = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        >>> timestamps = np.array([0, 1, 2])
        >>> index_data = tp.IndexData(features, timestamps)
        >>> len(index_data)
        3

        ```
    """

    features: List[np.ndarray]
    timestamps: np.ndarray

    def __init__(
        self,
        features: List[np.ndarray],
        timestamps: np.ndarray,
        schema: Optional[Schema] = None,
    ) -> None:
        """Initializes the IndexData object by checking and setting the features
        and timestamps.

        Raises:
            ValueError: If features are not one-dimensional arrays.
            ValueError: If the number of elements in features and timestamps
                do not match.
        """

        self.features = features
        self.timestamps = timestamps

        if schema is not None:
            self.check_schema(schema)

    def check_schema(self, schema: Schema):
        if not config.debug_mode:
            return

        # Check that the data (features & timestamps) matches the schema.

        if self.timestamps.ndim != 1:
            raise ValueError("timestamps must be one-dimensional arrays")

        if self.timestamps.dtype.type != np.float64:
            raise ValueError("Timestamps should be float64")

        if len(self.features) != len(schema.features):
            raise ValueError(
                "Wrong number of features. Event has "
                f"{len(self.features)} features while schema has "
                f"{len(schema.features)} features."
            )

        for feature_data, feature_schema in zip(self.features, schema.features):
            if feature_data.ndim != 1:
                raise ValueError("Features must be one-dimensional arrays")

            expected_numpy_type = _DTYPE_REVERSE_MAPPING[feature_schema.dtype]
            if feature_data.dtype.type != expected_numpy_type:
                raise ValueError(
                    "The schema does not match the feature dtype. Feature "
                    f"{feature_schema.name!r} has numpy dtype = "
                    f"{feature_data.dtype} but schema has temporian dtype = "
                    f"{feature_schema.dtype!r}. From the schema, the numpy"
                    f"type is expected to be {expected_numpy_type!r}."
                )

            if self.timestamps.shape != feature_data.shape:
                raise ValueError(
                    "The number of feature values for"
                    f" {feature_schema.name!r} ({len(feature_data)}) does not"
                    " match the number of timestamp values"
                    f" ({len(self.timestamps)})."
                )

    def __eq__(self, other) -> bool:
        if not isinstance(other, IndexData):
            return False

        if not np.array_equal(self.timestamps, other.timestamps):
            return False

        for f1, f2 in zip(self.features, other.features):
            if f1.dtype.kind == "f":
                if not np.allclose(f1, f2, equal_nan=True):
                    return False
            else:
                if not np.array_equal(f1, f2):
                    return False

        return True

    def __len__(self) -> int:
        """Number of events / timestamps."""

        return len(self.timestamps)


class EventSet(EventSetOperations):
    """Actual temporal data.

    Use [`tp.event_set()`][temporian.event_set] to create an EventSet manually,
    or [`tp.from_pandas()`][temporian.from_pandas] to create an EventSet from a
    pandas DataFrame.
    """

    # TODO: improve EventSet's docstring. Possibly link to Temporal data section
    # user guide, or explain briefly what the data's structure is.

    def __init__(
        self,
        data: Dict[NormalizedIndexKey, IndexData],
        schema: Schema,
        name: Optional[str] = None,
    ) -> None:
        self._data = data
        self._schema = schema
        self._name = name

        # EventSetNode created when "self.node()" is called.
        self._internal_node: Optional[EventSetNode] = None

    @property
    def data(self) -> Dict[NormalizedIndexKey, IndexData]:
        return self._data

    @property
    def schema(self) -> Schema:
        return self._schema

    @property
    def name(self) -> Optional[str]:
        return self._name

    @name.setter
    def name(self, name: Optional[str]) -> None:
        self._name = name

    def get_index_keys(self, sort: bool = False) -> List[NormalizedIndexKey]:
        idx_list = list(self.data.keys())
        return sorted(idx_list) if sort else idx_list

    def get_arbitrary_index_key(self) -> Optional[IndexKey]:
        """Gets an arbitrary index key.

        If the EventSet is empty, return None.
        """

        if self._data:
            return next(iter(self._data.keys()))
        return None

    def get_arbitrary_index_data(self) -> Optional[IndexData]:
        """Gets data from an arbitrary index key.

        If the EventSet is empty, return None.
        """

        if self._data:
            return next(iter(self._data.values()))
        return None

    def node(self, force_new_node: bool = False) -> EventSetNode:
        """Creates an [`EventSetNode`][temporian.EventSetNode] able to consume
        this EventSet.

        If called multiple times with `force_new_node=False` (default), the same
        node is returned.

        Usage example:
            ```python
            >>> my_evset = tp.event_set(
            ...     timestamps=[1, 2, 3, 4],
            ...     features={
            ...         "feature_1": [0.5, 0.6, np.nan, 0.9],
            ...         "feature_2": ["red", "blue", "red", "blue"],
            ...     },
            ... )
            >>> my_node = my_evset.node()

            ```

        Args:
            force_new_node: If false (default), return the same node each time
                `node` is called. If true, a new node is created each time.

        Returns:
            An EventSetNode able to consume this EventSet.
        """

        if self._internal_node is not None and not force_new_node:
            # "node" was already called. Return the cached node.
            return self._internal_node

        self._internal_node = create_node_with_new_reference(
            schema=self._schema,
            name=self._name,
        )
        return self._internal_node

    def get_index_value(
        self, index_key: IndexKey, normalize: bool = True
    ) -> IndexData:
        """Gets the value for a specified index key.

        The index key must be a tuple of values corresponding to the indexes
        of the EventSet.
        """

        if normalize:
            index_key = normalize_index_key(index_key)
        return self.data[index_key]

    def set_index_value(
        self, index_key: IndexKey, value: IndexData, normalize: bool = True
    ) -> None:
        """Sets the value for a specified index key.

        The index key must be a tuple of values corresponding to the indexes
        of the EventSet.
        """

        if normalize:
            index_key = normalize_index_key(index_key)
        self.data[index_key] = value

    def __eq__(self, other) -> bool:
        if not isinstance(other, EventSet):
            return False

        if self._name != other._name:
            return False

        if self._schema != other._schema:
            return False

        if self._data != other._data:
            return False

        return True

    def plot(self, *args, **wargs) -> Any:
        """Plots the EventSet. See [`tp.plot()`][temporian.plot] for details.

        Example usage:

            ```python
            >>> evset = tp.event_set(timestamps=[1, 2, 3], features={"f1": [0, 42, 10]})
            >>> evset.plot()

            ```
        """

        from temporian.implementation.numpy.data import plotter

        return plotter.plot(evsets=self, *args, **wargs)

    def __sizeof__(self) -> int:
        size = sys.getsizeof(self.data)
        for index_key, index_data in self.data.items():
            size += sys.getsizeof(index_key) + sys.getsizeof(
                index_data.timestamps
            )
            for feature in index_data.features:
                size += sys.getsizeof(feature)
        return size

    def memory_usage(self) -> int:
        """Gets the approximated memory usage of the EventSet in bytes.

        Takes into account garbage collector overhead.
        """

        return sys.getsizeof(self)

    def num_events(self) -> int:
        """Total number of events."""

        count = 0
        for data in self.data.values():
            count += len(data.timestamps)
        return count

    def check_same_sampling(self, other: EventSet):
        """Checks if two EventSets have the same sampling."""
        self.node().check_same_sampling(other.node())

    @property
    def creator(self) -> Optional[Operator]:
        """Creator.

        The creator is the operator that outputted this EventSet. Manually
        created EventSets have a `None` creator.
        """
        return self.node()._creator

    def __repr__(self) -> str:
        """Text representation, showing schema and data"""
        from temporian.implementation.numpy.data.display_utils import (
            display_text,
        )

        return display_text(self)

    def _repr_html_(self) -> str:
        """HTML representation, mainly for IPython notebooks."""
        from temporian.implementation.numpy.data.display_utils import (
            display_html,
        )

        return display_html(self)
