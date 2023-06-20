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

"""Set index operator class and public API function definition."""

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
        index_to_add: List[str],
    ) -> None:
        super().__init__()

        self._output_feature_schemas = self._get_output_feature_schemas(
            input, index_to_add
        )
        self._output_indexes = self._get_output_indexes(input, index_to_add)
        self._index_to_add = index_to_add

        self.add_input("input", input)

        self.add_attribute("index_to_add", index_to_add)

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
        self, input: Node, index_to_add: List[str]
    ) -> List[FeatureSchema]:
        return [
            feature
            for feature in input.schema.features
            if feature.name not in index_to_add
        ]

    def _get_output_indexes(
        self, input: Node, index_to_add: List[str]
    ) -> List[IndexSchema]:
        index_dict = input.schema.index_name_to_dtype()
        feature_dict = input.schema.feature_name_to_dtype()

        new_indexes: List[IndexSchema] = []
        for index_name in index_to_add:
            if index_name not in feature_dict:
                raise ValueError(
                    f"Add index {index_name} is not part of the features."
                )

            if index_name in index_dict:
                raise ValueError(
                    f"Add index {index_name} is already part of the index."
                )

            new_indexes.append(
                IndexSchema(name=index_name, dtype=feature_dict[index_name])
            )
        # Note: The new index is added after the existing index items.
        return input.schema.indexes + new_indexes

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="ADD_INDEX",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="index_to_add",
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
    def index_to_add(self) -> List[str]:
        return self._index_to_add


operator_lib.register_operator(AddIndexOperator)


def _normalize_index_to_add(
    index_names: Union[str, List[str]],
) -> List[str]:
    if isinstance(index_names, str):
        return [index_names]

    if len(index_names) == 0:
        raise ValueError("Cannot specify empty list as `index_names` argument.")

    return index_names


def _normalize_index_to_set(
    index_names: Union[str, List[str]],
) -> List[str]:
    if isinstance(index_names, str):
        return [index_names]

    if len(index_names) == 0:
        raise ValueError("Cannot specify empty list as `index_names` argument.")

    return index_names


def add_index(input: Node, index_to_add: Union[str, List[str]]) -> Node:
    """Adds one or more features as index in a node.

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
        input: Input node object for which the index is to be set or
            updated.
        index_to_add: List of feature names (strings) that should be used as
            the new index. These feature names should already exist in `input`.

    Returns:
         New node with the updated index.

    Raises:
        KeyError: If any of the specified `index_to_add` are not found in
            `input`.
    """

    index_to_add = _normalize_index_to_add(index_to_add)
    return AddIndexOperator(input, index_to_add).outputs["output"]


def set_index(input: Node, index: Union[str, List[str]]) -> Node:
    """Replaces the index in a node.

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
        ...     index_features=["store"]
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
        input: Input node object for which the index is to
            be set or updated.
        index: List of index / feature names (strings) used as
            the new index. These feature names should be either index or
            features in `input`.

    Returns:
        New node with the updated index.

    Raises:
        KeyError: If any of the specified `index_to_add` are not found in
            `input`.
    """

    new_index = _normalize_index_to_set(index)

    # Note
    # The set_index is implemented as a drop_index + add_index.
    # The implementation could be improved (simpoler, faster) with a new
    # operator to re-order the index items.

    if len(input.schema.indexes) != 0:
        input = drop_index(input)

    if len(new_index) != 0:
        input = add_index(input, new_index)

    return input
