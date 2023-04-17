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

"""DropIndex operator."""
from typing import List, Optional, Union

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class DropIndexOperator(Operator):
    def __init__(
        self,
        event: Event,
        index_to_drop: List[str],
        keep: bool,
    ) -> None:
        super().__init__()

        # "index_to_drop" is the list of indexes in "event" to drop. If "keep"
        # is true, those indexes will be converted into features.
        self._index_to_drop = index_to_drop
        self._keep = keep

        self.add_input("event", event)
        self.add_attribute("index_to_drop", index_to_drop)
        self.add_attribute("keep", keep)

        output_features = self._output_features(event, index_to_drop, keep)

        output_sampling = Sampling(
            index=[
                (index_name, index_dtype)
                for index_name, index_dtype in event.sampling.index_dtypes.items()
                if index_name not in index_to_drop
            ]
        )

        self.add_output(
            "event",
            Event(
                features=output_features,
                sampling=output_sampling,
                creator=self,
            ),
        )
        self.check()

    def _output_features(
        self, event: Event, index_to_drop: List[str], keep: bool
    ) -> List[Feature]:
        new_features = []
        if keep:
            # Convert the index to drop into features.
            #
            # Note: The new features are added first.

            new_features = []
            for index_name in index_to_drop:
                # check no other feature exists with this name
                if index_name in event.feature_names:
                    raise ValueError(
                        f"Feature name {index_name} coming from index already"
                        " exists in event."
                    )

                new_features.append(
                    Feature(
                        name=index_name,
                        dtype=event.sampling.index_dtypes[index_name],
                    )
                )

        existing_features = [
            Feature(name=feature.name, dtype=feature.dtype)
            for feature in event.features
        ]
        return new_features + existing_features

    def dst_feature_names(self) -> List[str]:
        feature_names = self.inputs["event"].feature_names
        if self._keep:
            return self._index_to_drop + feature_names
        else:
            return feature_names

    def dst_index_names(self) -> List[str]:
        return [
            index_level.name
            for index_level in self.inputs["event"].sampling.index
            if index_level.name not in self._index_to_drop
        ]

    @property
    def index_to_drop(self) -> List[str]:
        return self._index_to_drop

    @property
    def keep(self) -> bool:
        return self._keep

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="DROP_INDEX",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="index_names",
                    type=pb.OperatorDef.Attribute.Type.REPEATED_STRING,
                ),
                pb.OperatorDef.Attribute(
                    key="keep",
                    type=pb.OperatorDef.Attribute.Type.BOOL,
                ),
            ],
            inputs=[
                pb.OperatorDef.Input(key="event"),
            ],
            outputs=[pb.OperatorDef.Output(key="event")],
        )


operator_lib.register_operator(DropIndexOperator)


def _normalize_index_to_drop(
    event: Event, index_names: Optional[Union[List[str], str]]
) -> List[str]:
    if index_names is None:
        return event.sampling.index_names

    if isinstance(index_names, str):
        index_names = [index_names]

    if len(index_names) == 0:
        raise ValueError("Cannot specify empty list as `index_names` argument.")

    # check if any index names are missing from the index
    missing_index_names = [
        index_name
        for index_name in index_names
        if index_name not in event.sampling.index_names
    ]
    if missing_index_names:
        raise KeyError(
            f"Dropped indexes {missing_index_names} are missing from the"
            f" input index. The input index is {event.sampling.index}."
        )

    return index_names


def drop_index(
    event: Event,
    index_to_drop: Optional[Union[str, List[str]]] = None,
    keep: bool = True,
) -> Event:
    """
    Removes one or more index columns from the given `Event` object and returns
    a new `Event` object with the updated index. Optionally, it can also keep
    the removed index columns as features in the output `Event`.

    Args:
        event:
            The input `Event` object from which the specified index columns
            should be removed.
        index_to_drop:
            The index column(s) to be removed from the input `Event`.
            This can be a single column name (str) or a list of column names
            (List[str]). If not specified or set to `None`, all index columns in
            the input `Event` will be removed. Defaults to `None`.
        keep:
            A flag indicating whether the removed index columns should be kept
            as features in the output `Event`. Defaults to `True`.

    Returns:
        A new `Event` object with the updated index, where the specified index
        column(s) have been removed. If `keep` is set to `True`, the removed
        index columns will be included as features in the output `Event`.

    Raises:
        ValueError:
            If an empty list is provided as the `index_names` argument.
        KeyError:
            If any of the specified `index_names` are missing from the input
            `Event`'s index.
        ValueError:
            If a feature name coming from the index already exists in the input
            `Event`, and the `keep` flag is set to `True`.

    Examples:
        Given an input `Event` with index names ['A', 'B', 'C'] and features
        names ['X', 'Y', 'Z']:

        1. drop_index(event, index_names='A', keep=True)
           Output `Event` will have index names ['B', 'C'] and features names
           ['X', 'Y', 'Z', 'A'].

        2. drop_index(event, index_names=['A', 'B'], keep=False)
           Output `Event` will have index names ['C'] and features names
           ['X', 'Y', 'Z'].

        3. drop_index(event, index_names=None, keep=True)
           Output `Event` will have index names [] (empty index) and features
           names ['X', 'Y', 'Z', 'A', 'B', 'C'].
    """
    index_to_drop = _normalize_index_to_drop(event, index_to_drop)
    return DropIndexOperator(event, index_to_drop, keep).outputs["event"]
