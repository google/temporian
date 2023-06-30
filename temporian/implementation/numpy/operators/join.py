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


"""Implementation for the Join operator."""


from typing import Dict, Any
from dataclasses import dataclass

import numpy as np

from temporian.implementation.numpy.data.event_set import IndexData, EventSet
from temporian.core.operators.join import Join
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation
from temporian.implementation.numpy_cc.operators import operators_cc
from temporian.implementation.numpy.data.event_set import (
    IndexData,
    EventSet,
    tp_dtype_to_np_dtype,
)


@dataclass
class _OutputFeature:
    missing_value: Any
    np_dtype: Any
    feature_idx: int


class JoinNumpyImplementation(OperatorImplementation):
    def __init__(self, operator: Join) -> None:
        assert isinstance(operator, Join)
        super().__init__(operator)

    def __call__(self, left: EventSet, right: EventSet) -> Dict[str, EventSet]:
        assert isinstance(self.operator, Join)

        output_schema = self.output_schema("output")

        on = self.operator.on
        if on is not None:
            left_on_feature_idx = left.schema.feature_names().index(on)
            right_on_feature_idx = right.schema.feature_names().index(on)

        right_feature_defs = []
        for i, f in enumerate(right.schema.features):
            if on is not None and i == right_on_feature_idx:
                continue
            right_feature_defs.append(
                _OutputFeature(
                    missing_value=f.dtype.missing_value(),
                    np_dtype=tp_dtype_to_np_dtype(f.dtype),
                    feature_idx=i,
                )
            )

        # Create output EventSet
        output_evset = EventSet(data={}, schema=output_schema)

        # Fill output EventSet's data
        for index_key, left_item in left.data.items():
            num_output_events = len(left_item.timestamps)

            # The left features are passed directly
            dst_left_data = left_item.features

            # Initialize all the output right feature values as missing values
            dst_right_data = []
            for right_feature_def in right_feature_defs:
                dst_right_data.append(
                    np.full(
                        shape=num_output_events,
                        fill_value=right_feature_def.missing_value,
                        dtype=right_feature_def.np_dtype,
                    )
                )

            if index_key in right.data:
                right_item = right[index_key]

                if on is None:
                    join_idxs = operators_cc.left_join_idxs(
                        left_item.timestamps, right_item.timestamps
                    )
                else:
                    join_idxs = operators_cc.left_join_on_idxs(
                        left_item.timestamps,
                        right_item.timestamps,
                        left_item.features[left_on_feature_idx],
                        right_item.features[right_on_feature_idx],
                    )
                    print("@@@join_idxs (on):\n", join_idxs, flush=True)

                for dst_right_feature, right_feature_def in zip(
                    dst_right_data, right_feature_defs
                ):
                    np.putmask(
                        dst_right_feature,
                        np.not_equal(join_idxs, -1),
                        right_item.features[right_feature_def.feature_idx][
                            join_idxs
                        ],
                    )

            output_evset[index_key] = IndexData(
                dst_left_data + dst_right_data,
                left_item.timestamps,
                schema=output_schema,
            )

        return {"output": output_evset}


implementation_lib.register_operator_implementation(
    Join, JoinNumpyImplementation
)
