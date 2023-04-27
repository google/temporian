"""DropIndex operator."""
from typing import List, Optional, Union

from temporian.core import operator_lib
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators.base import Operator
from temporian.proto import core_pb2 as pb


class SetIndexOperator(Operator):
    """SetIndex operator."""

    def __init__(
        self,
        event: Event,
        feature_names: Optional[Union[List[str], str]] = None,
        append: bool = False,
    ) -> None:
        super().__init__()

        # process feature_names
        feature_names = self._process_feature_names(event, feature_names)

        # input event
        self.add_input("event", event)

        # attributes
        self.add_attribute("feature_names", feature_names)
        self.add_attribute("append", append)

        # output features
        output_features = self._generate_output_features(event, feature_names)

        # output sampling

        output_sampling = Sampling(
            index_levels=[
                (index_name, index_dtype)
                for index_name, index_dtype in event.sampling.index
            ]
            + [
                (index_name, event.dtypes[index_name])
                for index_name in feature_names
            ]
            if append
            else [
                (index_name, event.dtypes[index_name])
                for index_name in feature_names
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

    def _process_feature_names(
        self,
        event: Event,
        feature_names: Optional[Union[List[str], str]],
    ) -> List[str]:
        if isinstance(feature_names, str):
            feature_names = [feature_names]
        missing_feature_names = [
            label
            for label in feature_names
            if label not in [feature.name for feature in event.features]
        ]
        if missing_feature_names:
            raise KeyError(missing_feature_names)

        return feature_names

    def _generate_output_features(
        self, event: Event, feature_names: List[str]
    ) -> List[Feature]:
        output_features = [
            Feature(name=feature.name, dtype=feature.dtype)
            for feature in event.features
            if feature.name not in feature_names
        ]
        return output_features

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="SET_INDEX",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="feature_names",
                    type=pb.OperatorDef.Attribute.Type.REPEATED_STRING,
                ),
                pb.OperatorDef.Attribute(
                    key="append",
                    type=pb.OperatorDef.Attribute.Type.BOOL,
                ),
            ],
            inputs=[
                pb.OperatorDef.Input(key="event"),
            ],
            outputs=[pb.OperatorDef.Output(key="event")],
        )


operator_lib.register_operator(SetIndexOperator)


def set_index(
    event: Event, feature_names: List[str], append: bool = False
) -> Event:
    """Sets one or more index columns for the given `Event` object and returns
    a new `Event` object with the updated index.

    Optionally, it can also append the new index columns to the existing index.

    The input `event` object remains unchanged. The function returns a new
    event with the specified index changes.

    Examples:
        Given an input `Event` with index names ['A', 'B', 'C'] and features
        names ['X', 'Y', 'Z']:

        1. `set_index(event, feature_names=['X'], append=False)`
           Output `Event` will have index names ['X'] and features names
           ['Y', 'Z'].

        2. `set_index(event, feature_names=['X', 'Y'], append=False)`
           Output `Event` will have index names ['X', 'Y'] and
           features names ['Z'].

        3. `set_index(event, feature_names=['X', 'Y'], append=True)`
           Output `Event` will have index names ['A', 'B', 'C', 'X', 'Y'] and
           features names ['Z'].

    Args:
        event: Input `Event` object for which the index is to be set or
            updated.
        feature_names: List of feature names (strings) that should be used as
            the new index. These feature names should already exist in `event`.
        append: Flag indicating whether the new index should be appended to the
            existing index (True) or replace it (False). Defaults to `False`.

    Returns:
        New `Event` with the updated index, where the specified feature names have
        been set or appended as index columns, based on the `append` flag.

    Raises:
        KeyError: If any of the specified `feature_names` are not found in
            `event`.
    """
    return SetIndexOperator(event, feature_names, append).outputs["event"]
