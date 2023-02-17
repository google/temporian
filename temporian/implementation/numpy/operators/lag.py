from temporian.core.operators.lag import LagOperator
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.sampling import NumpySampling


class LagNumpyImplementation:
    def __init__(self, operator: LagOperator) -> None:
        super().__init__()
        self.operator = operator

    def __call__(self, event: NumpyEvent) -> NumpyEvent:
        duration = self.operator.attributes()["duration"]

        if duration < 0:
            raise ValueError("Duration must be positive")

        new_sampling = NumpySampling(
            data={},
            names=event.sampling.names,
        )
        output = NumpyEvent(data=event.data, sampling=new_sampling)

        for index, timestamps in event.sampling.data.items():
            new_sampling.data[index] = timestamps + duration
            output_data = output.data[index]
            for feature in output_data:
                feature.name = f"lag_{feature.name}"

        return {"event": output}
