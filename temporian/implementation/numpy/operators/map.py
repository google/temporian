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

"""Implementation for the Map operator."""

from inspect import signature
from typing import Dict

import numpy as np
from temporian.core.types import MapExtras
from temporian.implementation.numpy.data.dtype_normalization import (
    tp_dtype_to_np_dtype,
)

from temporian.implementation.numpy.data.event_set import IndexData, EventSet
from temporian.core.operators.map import Map
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation


class MapNumpyImplementation(OperatorImplementation):
    def __init__(self, operator: Map) -> None:
        assert isinstance(operator, Map)
        super().__init__(operator)

    def __call__(self, input: EventSet) -> Dict[str, EventSet]:
        assert isinstance(self.operator, Map)

        output_schema = self.output_schema("output")

        func = self.operator.func
        receive_extras = self.operator.receive_extras

        # Create output EventSet
        output_evset = EventSet(data={}, schema=output_schema)

        # Fill output EventSet's data
        for index_key, index_data in input.data.items():
            # Iterate over features and apply func
            features = []
            for feature_schema, orig_feature, output_dtype in zip(
                input.schema.features,
                index_data.features,
                output_schema.feature_dtypes(),
            ):
                extras = MapExtras(
                    index_key=index_key,
                    timestamp=0,
                    feature_name=feature_schema.name,
                )

                # TODO: preallocate numpy array directly when output dtype isn't
                # string (in which case we need to know the max length of func's
                # results before doing so)
                output_values = [None] * len(orig_feature)

                for i, (value, timestamp) in enumerate(
                    zip(orig_feature, index_data.timestamps)
                ):
                    extras.timestamp = timestamp
                    if receive_extras:
                        output_values[i] = func(value, extras)  # type: ignore
                    else:
                        output_values[i] = func(value)  # type: ignore

                try:
                    output_arr = np.array(
                        output_values, dtype=tp_dtype_to_np_dtype(output_dtype)
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"Failed to build array of type {output_dtype} with the"
                        " results of `func`. Make sure you are specifying the"
                        " correct `output_dypes` and returning those types in"
                        " `func`."
                    ) from exc

                features.append(output_arr)

            output_evset.set_index_value(
                index_key,
                IndexData(
                    features=features,
                    timestamps=index_data.timestamps,
                    schema=output_schema,
                ),
            )

        return {"output": output_evset}


implementation_lib.register_operator_implementation(Map, MapNumpyImplementation)
