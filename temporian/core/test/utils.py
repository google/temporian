from temporian.core import processor
from temporian.core.data import dtype
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators import base
from temporian.proto import core_pb2 as pb
from temporian.core import operator_lib

# The name of the operator is defined by the number of inputs and outputs.
# For example "OpI1O2" has 1 input and 2 outputs.


class OpO1(base.Operator):
    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="OpO1",
            place_holder=True,
            outputs=[pb.OperatorDef.Output(key="output")],
        )

    def __init__(self):
        super().__init__()
        sampling = Sampling(index=[], creator=self)
        self.add_output(
            "output",
            Event(
                features=[
                    Feature(
                        "f1", dtype.FLOAT64, sampling=sampling, creator=self
                    ),
                    Feature(
                        "f2", dtype.FLOAT64, sampling=sampling, creator=self
                    ),
                ],
                sampling=sampling,
            ),
        )
        self.check()


operator_lib.register_operator(OpO1)


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
                        "f1",
                        dtype.FLOAT64,
                        sampling=event.sampling(),
                        creator=self,
                    )
                ],
                sampling=event.sampling(),
            ),
        )
        self.check()


operator_lib.register_operator(OpI1O1)


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
                        "f1",
                        dtype.FLOAT64,
                        sampling=event_1.sampling(),
                        creator=self,
                    )
                ],
                sampling=event_1.sampling(),
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
                        sampling=event.sampling(),
                        creator=self,
                    )
                ],
                sampling=event.sampling(),
            ),
        )
        self.add_output(
            "output_2",
            Event(
                features=[
                    Feature(
                        "f1",
                        dtype.FLOAT64,
                        sampling=event.sampling(),
                        creator=self,
                    )
                ],
                sampling=event.sampling(),
            ),
        )
        self.check()


operator_lib.register_operator(OpI1O2)
