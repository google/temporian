from temporian.core.data.duration import duration_abbreviation
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

        sampling_data = {}
        output_data = {}

        prefix = "lag" if duration > 0 else "leak"
        duration_str = duration_abbreviation(duration)

        for index, timestamps in event.sampling.data.items():
            sampling_data[index] = timestamps + duration
            output_data[index] = []
            for feature in event.data[index]:
                new_feature = NumpyFeature(
                    data=feature.data,
                    name=f"{prefix}[{duration_str}]_{feature.name}",
                )
                output_data[index].append(new_feature)

        new_sampling = NumpySampling(
            data=sampling_data,
            index=event.sampling.index.copy(),
        )
        output_event = NumpyEvent(data=output_data, sampling=new_sampling)

        return {"event": output_event}
