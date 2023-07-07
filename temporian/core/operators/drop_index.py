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

"""Drop index operator class and public API function definition."""

from typing import List, Optional, Union

from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.node import (
    EventSetNode,
    create_node_new_features_new_sampling,
)
from temporian.core.data.schema import FeatureSchema, IndexSchema
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class DropIndexOperator(Operator):
    def __init__(
        self,
        input: EventSetNode,
        indexes: List[str],
        keep: bool,
    ) -> None:
        super().__init__()

        # `indexes`` is the list of indexes in `input` to drop. If
        # `keep` is true, those indexes will be converted into features.
        self._indexes = indexes
        self._keep = keep

        self.add_input("input", input)
        self.add_attribute("indexes", indexes)
        self.add_attribute("keep", keep)

        self._output_feature_schemas = self._get_output_feature_schemas(
            input, indexes, keep
        )

        self._output_indexes = [
            index_level
            for index_level in input.schema.indexes
            if index_level.name not in indexes
        ]

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
        self, input: EventSetNode, indexes: List[str], keep: bool
    ) -> List[FeatureSchema]:
        if not keep:
            return input.schema.features

        index_dict = input.schema.index_name_to_dtype()
        feature_dict = input.schema.feature_name_to_dtype()

        new_features: List[FeatureSchema] = []
        for index in indexes:
            if index not in index_dict:
                raise ValueError(f"{index} is not an index in input.")

            if index in feature_dict:
                raise ValueError(
                    f"{index} already exists in input's features. If you"
                    " want to drop the index, specify `keep=False`."
                )

            new_features.append(
                FeatureSchema(name=index, dtype=index_dict[index])
            )

        # Note: The new features are added after the existing features.
        return input.schema.features + new_features

    @property
    def output_feature_schemas(self) -> List[FeatureSchema]:
        return self._output_feature_schemas

    @property
    def output_indexes(self) -> List[IndexSchema]:
        return self._output_indexes

    @property
    def indexes(self) -> List[str]:
        return self._indexes

    @property
    def keep(self) -> bool:
        return self._keep

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="DROP_INDEX",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="indexes",
                    type=pb.OperatorDef.Attribute.Type.LIST_STRING,
                ),
                pb.OperatorDef.Attribute(
                    key="keep",
                    type=pb.OperatorDef.Attribute.Type.BOOL,
                ),
            ],
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(DropIndexOperator)


def _normalize_indexes(
    input: EventSetNode,
    indexes: Optional[Union[List[str], str]],
) -> List[str]:
    if indexes is None:
        # Drop all the indexes
        return input.schema.index_names()

    if isinstance(indexes, str):
        indexes = [indexes]

    if len(indexes) == 0:
        raise ValueError("Cannot specify empty list as `indexes` argument.")

    # Check that requested indexes exist
    index_dict = input.schema.index_name_to_dtype()
    for index in indexes:
        if index not in index_dict:
            raise KeyError(
                f"{index} is not an index in {input.schema.indexes}."
            )

    return indexes


@compile
def drop_index(
    input: EventSetNode,
    indexes: Optional[Union[str, List[str]]] = None,
    keep: bool = True,
) -> EventSetNode:
    """Removes indexes from an [`EventSetNode`][temporian.EventSetNode].

    Usage example:
        ```python
        >>> a = tp.event_set(
        ...     timestamps=[1, 2, 1, 0, 1, 1],
        ...     features={
        ...         "f1": [1, 1, 1, 2, 2, 2],
        ...         "f2": [1, 1, 2, 1, 1, 2],
        ...         "f3": [1, 1, 1, 1, 1, 1]
        ...     },
        ...     indexes=["f1", "f2"]
        ... )

        >>> # Both f1 and f2 are indices
        >>> a
        indexes: [('f1', int64), ('f2', int64)]
        features: [('f3', int64)]
        events:
            f1=2 f2=1 (2 events):
                timestamps: [0. 1.]
                'f3': [1 1]
            f1=1 f2=1 (2 events):
                timestamps: [1. 2.]
                'f3': [1 1]
            f1=1 f2=2 (1 events):
                timestamps: [1.]
                'f3': [1]
            f1=2 f2=2 (1 events):
                timestamps: [1.]
                'f3': [1]
        ...

        >>> # Drop "f2", remove it from features
        >>> b = tp.drop_index(a, "f2", keep=False)
        >>> b
        indexes: [('f1', int64)]
        features: [('f3', int64)]
        events:
            f1=2 (3 events):
                timestamps: [0. 1. 1.]
                'f3': [1 1 1]
            f1=1 (3 events):
                timestamps: [1. 1. 2.]
                'f3': [1 1 1]
        ...

        >>> # Drop both indices, keep them as features
        >>> b = tp.drop_index(a, ["f2", "f1"])
        >>> b
        indexes: []
        features: [('f3', int64), ('f2', int64), ('f1', int64)]
        events:
            (6 events):
                timestamps: [0. 1. 1. 1. 1. 2.]
                'f3': [1 1 1 1 1 1]
                'f2': [2 2 1 1 2 1]
                'f1': [1 1 1 2 2 1]
        ...

        ```

    Args:
        input: EventSetNode from which the specified indexes should be removed.
        indexes: Index column(s) to be removed from `input`. This can be a
            single column name (`str`) or a list of column names (`List[str]`).
            If not specified or set to `None`, all indexes in `input` will
            be removed. Defaults to `None`.
        keep: Flag indicating whether the removed indexes should be kept
            as features in the output `EventSetNode`. Defaults to `True`.

    Returns:
        New `EventSetNode` with the specified indexes removed. If `keep` is set to
        `True`, the removed indexes will be included as features in it.

    Raises:
        ValueError: If an empty list is provided as the `index_names` argument.
        KeyError: If any of the specified `index_names` are missing from
            `input`'s index.
        ValueError: If a feature name coming from the indexes already exists in
            `input`, and the `keep` flag is set to `True`.
    """
    indexes = _normalize_indexes(input, indexes)
    return DropIndexOperator(input, indexes, keep).outputs["output"]
