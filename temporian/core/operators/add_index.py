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

"""Add and set index operators classes and public API function definitions."""

from typing import List, Union

from temporian.core import operator_lib
from temporian.core.data.node import Node, create_node_new_features_new_sampling
from temporian.core.data.schema import FeatureSchema, IndexSchema
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb
from temporian.core.operators.drop_index import drop_index


class AddIndexOperator(Operator):
    def __init__(
        self,
        input: Node,
        indexes: List[str],
    ) -> None:
        super().__init__()

        self._output_feature_schemas = self._get_output_feature_schemas(
            input, indexes
        )
        self._output_indexes = self._get_output_indexes(input, indexes)
        self._indexes = indexes

        self.add_input("input", input)

        self.add_attribute("indexes", indexes)

        self.add_output(
            "output",
            create_node_new_features_new_sampling(
                features=self._output_feature_schemas,
                indexes=self._output_indexes,
                is_unix_timestamp=input.schema.is_unix_timestamp,
                creator=self,
            ),
        )
        self.check()

    def _get_output_feature_schemas(
        self, input: Node, indexes: List[str]
    ) -> List[FeatureSchema]:
        return [
            feature
            for feature in input.schema.features
            if feature.name not in indexes
        ]

    def _get_output_indexes(
        self, input: Node, indexes: List[str]
    ) -> List[IndexSchema]:
        index_dict = input.schema.index_name_to_dtype()
        feature_dict = input.schema.feature_name_to_dtype()

        new_indexes: List[IndexSchema] = []
        for index in indexes:
            if index not in feature_dict:
                raise ValueError(f"{index} is not a feature in input.")

            if index in index_dict:
                raise ValueError(f"{index} is already an index in input.")

            new_indexes.append(
                IndexSchema(name=index, dtype=feature_dict[index])
            )
        # Note: The new indexes are added after the existing ones.
        return input.schema.indexes + new_indexes

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="ADD_INDEX",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="indexes",
                    type=pb.OperatorDef.Attribute.Type.LIST_STRING,
                ),
            ],
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )

    @property
    def output_feature_schemas(self) -> List[FeatureSchema]:
        return self._output_feature_schemas

    @property
    def output_indexes(self) -> List[IndexSchema]:
        return self._output_indexes

    @property
    def indexes(self) -> List[str]:
        return self._indexes


operator_lib.register_operator(AddIndexOperator)


def _normalize_indexes(
    indexes: Union[str, List[str]],
) -> List[str]:
    if isinstance(indexes, str):
        return [indexes]

    if len(indexes) == 0:
        raise ValueError("Cannot specify empty list as `indexes` argument.")

    return indexes


def _normalize_indexes_to_set(
    indexes: Union[str, List[str]],
) -> List[str]:
    if isinstance(indexes, str):
        return [indexes]

    if len(indexes) == 0:
        raise ValueError("Cannot specify empty list as `indexes` argument.")

    return indexes


def add_index(input: Node, indexes: Union[str, List[str]]) -> Node:
    """Adds indexes to a [`Node`][temporian.Node].

    Usage example:
        ```python
        >>> a_evset = tp.event_set(
        ...     timestamps=[1, 2, 1, 0, 1, 1],
        ...     features={
        ...         "store": [1, 1, 1, 2, 2, 2],
        ...         "product": [1, 1, 2, 1, 1, 2],
        ...         "sales": [1, 1, 1, 1, 1, 1]
        ...     },
        ... )

        >>> # No index, store and product are regularfeatures
        >>> a_evset
        indexes: []
        features: [('store', int64), ('product', int64), ('sales', int64)]
        events:
            (6 events):
                timestamps: [0. 1. 1. 1. 1. 2.]
                'store': [2 1 1 2 2 1]
                'product': [1 1 2 1 2 1]
                'sales': [1 1 1 1 1 1]
        ...

        >>> # Add only "store" as index
        >>> a = a_evset.node()
        >>> result = tp.add_index(a, "store")
        >>> result.evaluate({a: a_evset})
        indexes: [('store', int64)]
        features: [('product', int64), ('sales', int64)]
        events:
            store=2 (3 events):
                timestamps: [0. 1. 1.]
                'product': [1 1 2]
                'sales': [1 1 1]
            store=1 (3 events):
                timestamps: [1. 1. 2.]
                'product': [1 2 1]
                'sales': [1 1 1]
        ...

        >>> # Add "store" and "product" as indices
        >>> result = tp.add_index(a, ["store", "product"])
        >>> result.evaluate({a: a_evset})
        indexes: [('store', int64), ('product', int64)]
        features: [('sales', int64)]
        events:
            store=2 product=1 (2 events):
                timestamps: [0. 1.]
                'sales': [1 1]
            store=1 product=1 (2 events):
                timestamps: [1. 2.]
                'sales': [1 1]
            store=1 product=2 (1 events):
                timestamps: [1.]
                'sales': [1]
            store=2 product=2 (1 events):
                timestamps: [1.]
                'sales': [1]
        ...

        ```

    Args:
        input: Node for which the indexes are to be set or updated.
        indexes: List of feature names (strings) that should be added to the
            indexes. These feature names should already exist in `input`.

    Returns:
        New Node with the extended index.

    Raises:
        KeyError: If any of the specified `indexes` are not found in `input`.
    """

    indexes = _normalize_indexes(indexes)
    return AddIndexOperator(input, indexes).outputs["output"]


def set_index(input: Node, indexes: Union[str, List[str]]) -> Node:
    """Replaces the index in a [`Node`][temporian.Node].

    This function is implemented as [`tp.drop_index()`](../drop_index)
    + [`tp.add_index()`](../add_index).

    Usage example:
        ```python
        >>> a_evset = tp.event_set(
        ...     timestamps=[1, 2, 1, 0, 1, 1],
        ...     features={
        ...         "store": [1, 1, 1, 2, 2, 2],
        ...         "product": [1, 1, 2, 1, 1, 2],
        ...         "sales": [1, 1, 1, 1, 1, 1]
        ...     },
        ...     indexes=["store"]
        ... )
        >>> a = a_evset.node()

        >>> # "store" is the current index
        >>> a_evset
        indexes: [('store', int64)]
        features: [('product', int64), ('sales', int64)]
        events:
            store=2 (3 events):
                timestamps: [0. 1. 1.]
                'product': [1 1 2]
                'sales': [1 1 1]
            store=1 (3 events):
                timestamps: [1. 1. 2.]
                'product': [1 2 1]
                'sales': [1 1 1]
        ...

        >>> # Set "product" as the only index, remove store
        >>> result = tp.set_index(a, "product")
        >>> result.evaluate({a: a_evset})
        indexes: [('product', int64)]
        features: [('sales', int64), ('store', int64)]
        events:
            product=1 (4 events):
                timestamps: [0. 1. 1. 2.]
                'sales': [1 1 1 1]
                'store': [2 2 1 1]
            product=2 (2 events):
                timestamps: [1. 1.]
                'sales': [1 1]
                'store': [2 1]
        ...

        >>> # Set both "store" and "product" as indices
        >>> result = tp.set_index(a, ["store", "product"])
        >>> result.evaluate({a: a_evset})
        indexes: [('store', int64), ('product', int64)]
        features: [('sales', int64)]
        events:
            store=2 product=1 (2 events):
                timestamps: [0. 1.]
                'sales': [1 1]
            store=2 product=2 (1 events):
                timestamps: [1.]
                'sales': [1]
            store=1 product=1 (2 events):
                timestamps: [1. 2.]
                'sales': [1 1]
            store=1 product=2 (1 events):
                timestamps: [1.]
                'sales': [1]
        ...

        ```

    Args:
        input: Node for which the indexes are to be set.
        indexes: List of index / feature names (strings) used as
            the new indexes. These names should be either indexes or features in
            `input`.

    Returns:
        New `Node` with the updated indexes.

    Raises:
        KeyError: If any of the specified `indexes` are not found in `input`.
    """

    indexes = _normalize_indexes_to_set(indexes)

    # Note
    # The set_index is implemented as a drop_index + add_index.
    # The implementation could be improved (simpler, faster) with a new
    # operator to re-order the indexes.

    if len(input.schema.indexes) != 0:
        input = drop_index(input)

    if len(indexes) != 0:
        input = add_index(input, indexes)

    return input
