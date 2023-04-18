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
        index_names: List[str],
        keep: bool,
    ) -> None:
        super().__init__()

        self._index_names = index_names
        self._keep = keep

        # input event
        self.add_input("event", event)

        # attributes
        self.add_attribute("index_names", index_names)
        self.add_attribute("keep", keep)

        # output features
        output_features = self._generate_output_features(
            event, index_names, keep
        )
        # output sampling
        output_sampling = Sampling(
            index_levels=[
                (index_name, index_dtype)
                for index_name, index_dtype in event.sampling.index
                if index_name not in index_names
            ]
        )
        # output event
        self.add_output(
            "event",
            Event(
                features=output_features,
                sampling=output_sampling,
                creator=self,
            ),
        )
        self.check()

    def _generate_output_features(
        self, event: Event, index_names: List[str], keep: bool
    ) -> List[Feature]:
        output_features = [
            Feature(name=feature.name, dtype=feature.dtype)
            for feature in event.features
        ]
        if keep:
            output_features = ([None] * len(index_names)) + output_features
            for i, index_name in enumerate(index_names):
                # check no other feature exists with this name
                if index_name in event.feature_names:
                    raise ValueError(
                        f"Feature name {index_name} coming from index already"
                        " exists in event."
                    )  # TODO: add automatic suffix instead of raising error? add capability to rename index

                output_features[i] = Feature(
                    name=index_name,
                    dtype=event.sampling.index.dtypes[
                        event.sampling.index.names.index(index_name)
                    ],
                )

        return output_features

    @property
    def dst_feat_names(self) -> List[str]:
        feature_names = self.inputs["event"].feature_names
        return (
            self._index_names + feature_names if self._keep else feature_names
        )

    @property
    def index_names(self) -> List[str]:
        return self._index_names

    @property
    def keep(self) -> bool:
        return self._keep

    def dst_index_names(self) -> List[str]:
        return [
            index_lvl_name
            for index_lvl_name in self.inputs["event"].sampling.index.names
            if index_lvl_name not in self._index_names
        ]

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


def _process_index_names(
    event: Event, index_names: Optional[Union[List[str], str]]
) -> List[str]:
    if index_names is None:
        return event.sampling.index

    if isinstance(index_names, str):
        index_names = [index_names]

    if len(index_names) == 0:
        raise ValueError("Cannot specify empty list as `index_names` argument.")

    # check if any index names are missing from the index
    missing_index_names = [
        index_name
        for index_name in index_names
        if index_name not in event.sampling.index
    ]
    if missing_index_names:
        raise KeyError(
            f"{missing_index_names} are missing from the input event's index."
        )

    return index_names


def drop_index(
    event: Event,
    index_names: Optional[Union[str, List[str]]] = None,
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
        index_names:
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
    index_names = _process_index_names(event, index_names)
    return DropIndexOperator(event, index_names, keep).outputs["event"]
