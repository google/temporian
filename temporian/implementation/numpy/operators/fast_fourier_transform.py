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


"""Implementation for the FFT operator."""


from typing import Dict
import numpy as np

from temporian.implementation.numpy.data.event_set import IndexData, EventSet
from temporian.core.operators.fast_fourier_transform import FastFourierTransform
from temporian.implementation.numpy import implementation_lib
from temporian.implementation.numpy.operators.base import OperatorImplementation


class FastFourierTransformNumpyImplementation(OperatorImplementation):
    def __init__(self, operator: FastFourierTransform) -> None:
        assert isinstance(operator, FastFourierTransform)
        super().__init__(operator)

    def __call__(self, input: EventSet) -> Dict[str, EventSet]:
        assert isinstance(self.operator, FastFourierTransform)

        output_schema = self.output_schema("output")

        # Create output EventSet
        output_evset = EventSet(data={}, schema=output_schema)

        num_events = self.operator.num_events
        num_output_features = self.operator.num_output_features
        num_spectral_lines = self.operator.num_spectral_lines
        hop_size = self.operator.hop_size

        if num_spectral_lines is None:
            num_spectral_lines = num_output_features
        else:
            assert num_spectral_lines <= num_output_features

        if self.operator.window is None:
            window = None
        elif self.operator.window == "hamming":
            window = np.hamming(num_events)
        else:
            raise ValueError(f"Unknown window {self.operator.window}")

        # Fill output EventSet's data
        for index_key, index_data in input.data.items():
            src_values = index_data.features[0]
            dst_timestamps = index_data.timestamps[(num_events - 1) :: hop_size]

            # TODO: Implement in c++.

            dst_values = []
            for evt_idx in range(num_events - 1, len(src_values), hop_size):
                start_idx = evt_idx - num_events + 1
                end_idx = evt_idx + 1
                selected_src_values = src_values[start_idx:end_idx]
                if window is not None:
                    selected_src_values = selected_src_values * window
                fft_res = np.fft.fft(selected_src_values)[:num_spectral_lines]
                ft_ampl = np.abs(fft_res).astype(src_values.dtype)
                dst_values.append(ft_ampl)
            dst_values = np.stack(dst_values, axis=1)

            output_evset.set_index_value(
                index_key,
                IndexData(
                    features=dst_values,
                    timestamps=dst_timestamps,
                    schema=output_schema,
                ),
            )

        return {"output": output_evset}


implementation_lib.register_operator_implementation(
    FastFourierTransform, FastFourierTransformNumpyImplementation
)
