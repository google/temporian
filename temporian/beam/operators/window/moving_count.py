# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Type
import apache_beam as beam
from temporian.beam.operators.window.base import BaseWindowBeamImplementation

from temporian.core.operators.window.moving_count import (
    MovingCountOperator,
)
from temporian.beam import implementation_lib
from temporian.implementation.numpy.operators.window.base import (
    BaseWindowNumpyImplementation,
)
from temporian.implementation.numpy.operators.window.moving_count import (
    MovingCountNumpyImplementation,
)


from abc import abstractmethod
from typing import Dict, Optional, Type
from temporian.core.operators.window.base import BaseWindowOperator

from temporian.implementation.numpy.operators.window.base import (
    BaseWindowNumpyImplementation,
)
from temporian.beam.operators.base import (
    BeamOperatorImplementation,
    beam_eventset_map,
    beam_eventset_map_with_sampling,
)
from temporian.beam.typing import (
    BeamEventSet,
    FeatureItem,
    BeamIndexKey,
    FeatureItemValue,
)


class MovingCountBeamImplementation(BaseWindowBeamImplementation):
    def _implementation(self) -> Type[BaseWindowNumpyImplementation]:
        return MovingCountNumpyImplementation

    def call(
        self, input: BeamEventSet, sampling: Optional[BeamEventSet] = None
    ) -> Dict[str, BeamEventSet]:
        assert isinstance(self.operator, BaseWindowOperator)

        numpy_implementation = self._implementation()(self.operator)

        if self.operator.has_sampling:

            def _run_with_sampling(
                index: BeamIndexKey,
                feature: Optional[FeatureItemValue],
                sampling: FeatureItemValue,
                feature_idx: int,
            ) -> FeatureItem:
                sampling_timestamps, _ = sampling
                output_values = (
                    numpy_implementation.apply_feature_wise_with_sampling(
                        src_timestamps=(
                            feature[0] if feature is not None else None
                        ),
                        src_feature=feature[1] if feature is not None else None,
                        sampling_timestamps=sampling_timestamps,
                        feature_idx=feature_idx,
                    )
                )
                return index, (sampling_timestamps, output_values)

            assert sampling is not None
            output = beam_eventset_map_with_sampling(
                input,
                sampling,
                name=f"{self.operator}",
                fn=_run_with_sampling,
            )

        else:

            def _run_without_sampling(
                item: FeatureItem,
                feature_idx: int,
            ) -> FeatureItem:
                indexes, (timestamps, input_values) = item
                print(
                    "calling _run_without_sampling with indexes",
                    indexes,
                    "feature_idx",
                    feature_idx,
                )
                output_values = numpy_implementation.apply_feature_wise(
                    src_timestamps=timestamps,
                    src_feature=input_values,
                    feature_idx=feature_idx,
                )
                return indexes, (timestamps, output_values)

            output = beam_eventset_map(
                input, name=f"{self.operator}", fn=_run_without_sampling
            )
            # output | beam.Map(lambda x: print("xxxx", x))
            for i, o in enumerate(output):
                o | "map" + str(i) >> beam.Map(lambda x: print("xxxx", x))
            # output | beam.CoGroupByKey() | beam.Map(lambda x: print("xxxx", x))
            # output | beam.CoGroupByKey() | beam.Filter(
            #     lambda x: True if x[0] == 0 else False
            # ) | beam.Map(lambda x: print("xxxx", x))
            # keep only first item in each key
            # output | "cogroup" >> beam.CoGroupByKey() | beam.ParDo(
            #     UngroupElements()
            # ) | beam.Map(lambda x: print("wwww", x))
            output | "cogroup" >> beam.CoGroupByKey() | beam.Map(
                lambda x: print("wwww", x)
            )
            output | "cogroupp" >> beam.CoGroupByKey() | beam.ParDo(
                UngroupElements()
            ) | beam.Map(lambda x: print("eeee", x))
            output = (
                output
                | "cogrouppp" >> beam.CoGroupByKey()
                | "pardo" >> beam.ParDo(UngroupElements())
            )
            # output = (
            #     output
            #     | "cogroupp" >> beam.CoGroupByKey()
            #     # | "filter"
            #     # >> beam.Filter(lambda x: True if x[0] == 0 else False)
            #     | beam.ParDo(UngroupElements())
            #     | beam.Map(lambda x: print("eeee", x))
            # )

        print("bbb output:", output)

        return {"output": output}


class UngroupElements(beam.DoFn):
    def process(self, element):
        k, v = element
        yield k, v[0]
        # for elem in list(v):
        # process your element
        # yield k, v


implementation_lib.register_operator_implementation(
    MovingCountOperator, MovingCountBeamImplementation
)
