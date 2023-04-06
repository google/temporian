"""Utilities for unit testing."""

import pandas as pd

from temporian.core.data import dtype
from temporian.core.data import event as event_lib
from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators import base
from temporian.proto import core_pb2 as pb
from temporian.core import operator_lib
from temporian.implementation.numpy.data.event import NumpyEvent

# The name of the operator is defined by the number of inputs and outputs.
# For example "OpI1O2" has 1 input and 2 outputs.

Event = event_lib.Event


def create_input_event():
    return event_lib.input_event(
        features=[
            Feature("f1", dtype.FLOAT32),
            Feature("f2", dtype.FLOAT32),
        ]
    )


def create_input_event_data():
    return NumpyEvent.from_dataframe(
        pd.DataFrame(
            {
                "timestamp": [0, 2, 4, 6],
                "f1": [1.0, 2.0, 3.0, 4.0],
                "f2": [5.0, 6.0, 7.0, 8.0],
            }
        )
    )


class OpI1O1(base.Operator):
    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="OpI1O1",
            inputs=[pb.OperatorDef.Input(key="event")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )

    def __init__(self, event: Event):
        super().__init__()

        self.add_input("event", event)
        self.add_output(
            "output",
            Event(
                features=[
                    Feature(
                        "f3",
                        dtype.FLOAT64,
                        sampling=event.sampling,
                        creator=self,
                    ),
                    Feature(
                        "f4",
                        dtype.FLOAT64,
                        sampling=event.sampling,
                        creator=self,
                    ),
                ],
                sampling=Sampling(index=[], creator=self),
                creator=self,
            ),
        )
        self.check()


operator_lib.register_operator(OpI1O1)


class OpI1O1NotCreator(base.Operator):
    """Unlike OpI1O1, OpI1O1NotCreator only passes the features/sampling."""

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="OpI1O1NotCreator",
            inputs=[pb.OperatorDef.Input(key="event")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )

    def __init__(self, event: Event):
        super().__init__()
        self.add_input("event", event)
        self.add_output(
            "output",
            Event(
                features=[f for f in event.features],
                sampling=event.sampling,
                creator=self,
            ),
        )
        self.check()


operator_lib.register_operator(OpI1O1NotCreator)


class OpI2O1(base.Operator):
    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="OpI2O1",
            inputs=[
                pb.OperatorDef.Input(key="event_1"),
                pb.OperatorDef.Input(key="event_2"),
            ],
            outputs=[pb.OperatorDef.Output(key="output")],
        )

    def __init__(self, event_1: Event, event_2: Event):
        super().__init__()
        self.add_input("event_1", event_1)
        self.add_input("event_2", event_2)
        self.add_output(
            "output",
            Event(
                features=[
                    Feature(
                        "f5",
                        dtype.FLOAT64,
                        sampling=event_1.sampling,
                        creator=self,
                    ),
                    Feature(
                        "f6",
                        dtype.FLOAT64,
                        sampling=event_1.sampling,
                        creator=self,
                    ),
                ],
                sampling=event_1.sampling,
                creator=self,
            ),
        )
        self.check()


operator_lib.register_operator(OpI2O1)


class OpI1O2(base.Operator):
    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="OpI1O2",
            inputs=[
                pb.OperatorDef.Input(key="event"),
            ],
            outputs=[
                pb.OperatorDef.Output(key="output_1"),
                pb.OperatorDef.Output(key="output_2"),
            ],
        )

    def __init__(self, event: Event):
        super().__init__()
        self.add_input("event", event)
        self.add_output(
            "output_1",
            Event(
                features=[
                    Feature(
                        "f1",
                        dtype.FLOAT64,
                        sampling=event.sampling,
                        creator=self,
                    )
                ],
                sampling=event.sampling,
                creator=self,
            ),
        )
        self.add_output(
            "output_2",
            Event(
                features=[
                    Feature(
                        "f1",
                        dtype.FLOAT64,
                        sampling=event.sampling,
                        creator=self,
                    )
                ],
                sampling=event.sampling,
                creator=self,
            ),
        )
        self.check()


operator_lib.register_operator(OpI1O2)
