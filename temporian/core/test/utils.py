from temporian.core import processor
from temporian.core.data import dtype
from temporian.core.data.event import Event
from temporian.core.data.feature import Feature
from temporian.core.data.sampling import Sampling
from temporian.core.operators import base
from temporian.proto import core_pb2 as pb

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
        sampling = Sampling(index=[])
        self.add_output(
            "output",
            Event(
                features=[
                    Feature("f1", dtype.FLOAT, sampling=sampling, creator=self),
                    Feature("f2", dtype.FLOAT, sampling=sampling, creator=self),
                ],
                sampling=sampling,
            ),
        )
        self.check()

    def _get_pandas_implementation(self):
        raise NotImplementedError()


class OpI1O1(base.Operator):
    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="OpI1O1",
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )

    def __init__(self, data: Event):
        super().__init__()
        self.add_input("input", data)
        self.add_output(
            "output",
            Event(
                features=[
                    Feature(
                        "f1",
                        dtype.FLOAT,
                        sampling=data.sampling(),
                        creator=self,
                    )
                ],
                sampling=data.sampling(),
            ),
        )
        self.check()

    def _get_pandas_implementation(self):
        raise NotImplementedError()


class OpI2O1(base.Operator):
    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="OpI2O1",
            inputs=[
                pb.OperatorDef.Input(key="input_1"),
                pb.OperatorDef.Input(key="input_2"),
            ],
            outputs=[pb.OperatorDef.Output(key="output")],
        )

    def __init__(self, input_1: Event, input_2: Event):
        super().__init__()
        self.add_input("input_1", input_1)
        self.add_input("input_2", input_2)
        self.add_output(
            "output",
            Event(
                features=[
                    Feature(
                        "f1",
                        dtype.FLOAT,
                        sampling=input_1.sampling(),
                        creator=self,
                    )
                ],
                sampling=input_1.sampling(),
            ),
        )
        self.check()

    def _get_pandas_implementation(self):
        raise NotImplementedError()


class OpI1O2(base.Operator):
    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="OpI1O2",
            inputs=[
                pb.OperatorDef.Input(key="input"),
            ],
            outputs=[
                pb.OperatorDef.Output(key="output_1"),
                pb.OperatorDef.Output(key="output_2"),
            ],
        )

    def __init__(self, data: Event):
        super().__init__()
        self.add_input("input", data)
        self.add_output(
            "output_1",
            Event(
                features=[
                    Feature(
                        "f1",
                        dtype.FLOAT,
                        sampling=data.sampling(),
                        creator=self,
                    )
                ],
                sampling=data.sampling(),
            ),
        )
        self.add_output(
            "output_2",
            Event(
                features=[
                    Feature(
                        "f1",
                        dtype.FLOAT,
                        sampling=data.sampling(),
                        creator=self,
                    )
                ],
                sampling=data.sampling(),
            ),
        )
        self.check()

    def _get_pandas_implementation(self):
        raise NotImplementedError()
