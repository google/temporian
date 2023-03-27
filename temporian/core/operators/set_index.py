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
        labels: Optional[Union[List[str], str]] = None,
        append: bool = False,
    ) -> None:
        super().__init__()

        # process labels
        labels = self._process_labels(event, labels)

        # input event
        self.add_input("event", event)

        # attributes
        self.add_attribute("labels", labels)
        self.add_attribute("append", append)

        # output features
        output_features = self._generate_output_features(event, labels)

        # output sampling
        output_sampling = Sampling(
            index=[
                index_name for index_name in event.sampling().index() + labels
            ]
            if append
            else labels
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

    def _process_labels(
        self,
        event: Event,
        labels: Optional[Union[List[str], str]],
    ) -> List[str]:
        if isinstance(labels, str):
            labels = [labels]
        missing_labels = [
            label
            for label in labels
            if label not in [feature.name() for feature in event.features()]
        ]
        if missing_labels:
            raise KeyError(missing_labels)

        return labels

    def _generate_output_features(
        self, event: Event, labels: List[str]
    ) -> List[Feature]:
        output_features = [
            Feature(name=feature.name(), dtype=feature.dtype())
            for feature in event.features()
            if feature.name not in labels
        ]
        return output_features

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="SET_INDEX",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="labels",
                    type=pb.OperatorDef.Attribute.Type.REPEATED_STRING,
                    is_optional=False,
                ),
                pb.OperatorDef.Attribute(
                    key="append",
                    type=pb.OperatorDef.Attribute.Type.BOOL,
                    is_optional=False,
                ),
            ],
            inputs=[
                pb.OperatorDef.Input(key="event"),
            ],
            outputs=[pb.OperatorDef.Output(key="event")],
        )


operator_lib.register_operator(SetIndexOperator)


def set_index(
    event: Event, labels: Optional[List[str]] = None, append: bool = True
) -> Event:
    return SetIndexOperator(event, labels, append).outputs()["event"]
