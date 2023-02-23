from temporian.core.operators.lag import LagOperator
from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling


class LagNumpyImplementation:
    def __init__(self, operator: LagOperator) -> None:
        super().__init__()
        self.operator = operator

    def __call__(self, event: NumpyEvent) -> NumpyEvent:
        duration = self.operator.attributes()["duration"]

        new_sampling = NumpySampling(
            data={},
            index=event.sampling.index.copy(),
        )
        output_event = NumpyEvent(data={}, sampling=new_sampling)

        for index, timestamps in event.sampling.data.items():
            new_sampling.data[index] = timestamps + duration
            output_event.data[index] = []
            output_data = output_event.data[index]
            for feature in event.data[index]:
                new_feature = NumpyFeature(
                    data=feature.data.copy(),
                    name=f"lag_{feature.name}",
                )
                output_data.append(new_feature)

        return {"event": output_event}
