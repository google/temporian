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
            index=[
                (index_name, index_dtype)
                for index_name, index_dtype in event.sampling.index_dtypes.items()
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
    return SetIndexOperator(event, feature_names, append).outputs["event"]
