from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.sampling import NumpySampling


class NumpyLagOperator:
    def __init__(self, duration: float) -> None:
        super().__init__()
        self.duration = duration

    def __call__(self, event: NumpyEvent) -> NumpyEvent:
        new_sampling = NumpySampling(
            data={},
            names=event.sampling.names,
        )
        output = NumpyEvent(data=event.data, sampling=new_sampling)

        for index, timestamps in event.sampling.data.items():
            new_sampling.data[index] = timestamps + self.duration
            for feature in output.data[index]:
                feature.name = f"lag_{feature.name}"

        return {"event": output}
