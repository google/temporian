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


"""FFT operator class and public API function definitions."""

from typing import Optional
from temporian.core import operator_lib
from temporian.core.compilation import compile
from temporian.core.data.node import (
    EventSetNode,
    create_node_new_features_new_sampling,
)
from temporian.core.operators.base import Operator
from temporian.core.typing import EventSetOrNode
from temporian.proto import core_pb2 as pb
from temporian.utils.rtcheck import rtcheck
from temporian.core.data.dtype import DType


_WINDOWS = {"hamming"}


class FastFourierTransform(Operator):
    def __init__(
        self,
        input: EventSetNode,
        num_events: int,
        hop_size: int,
        window: Optional[str] = None,
        num_spectral_lines: Optional[int] = None,
    ):
        super().__init__()

        # Save attributes
        self._num_events = num_events
        self._window = window
        self._num_spectral_lines = num_spectral_lines
        self._hop_size = hop_size

        self.add_attribute("num_events", num_events)
        self.add_attribute("hop_size", hop_size)

        if window is not None:
            self.add_attribute("window", window)
        if num_spectral_lines is not None:
            self.add_attribute("num_spectral_lines", num_spectral_lines)

        # Check errors
        if num_events <= 1:
            raise ValueError(
                "num_events should be strictly positive. Got"
                f" {num_events} instead."
            )
        if len(input.features) != 1:
            raise ValueError(
                "FFT input needs to be a single feature. Got"
                f" {len(input.features)} features instead."
            )
        if input.features[0].dtype not in [DType.FLOAT32, DType.FLOAT64]:
            raise ValueError(
                "FFT input feature should be tp.float32 or tp.float64. Got"
                f" {input.features[0].dtype} instead."
            )
        if window is not None and window not in _WINDOWS:
            raise ValueError(
                f"window should be None or in {_WINDOWS}. Got {window} instead."
            )
        if num_spectral_lines is not None:
            if num_spectral_lines <= 0:
                raise ValueError(
                    "num_spectral_lines should be positive. Got"
                    f" {num_spectral_lines} instead"
                )
            if num_spectral_lines > num_events // 2:
                raise ValueError(
                    "num_spectral_lines should be less or equal to num_events"
                    f" // 2. Got {num_spectral_lines} instead"
                )

        self.add_input("input", input)
        self.add_output(
            "output",
            create_node_new_features_new_sampling(
                features=[
                    (f"a{i}", input.schema.features[0].dtype)
                    for i in range(self.num_output_features)
                ],
                indexes=input.schema.indexes,
                is_unix_timestamp=input.schema.is_unix_timestamp,
                creator=self,
            ),
        )

        self.check()

    @property
    def num_spectral_lines(self) -> Optional[int]:
        return self._num_spectral_lines

    @property
    def num_output_features(self) -> int:
        if self._num_spectral_lines is None:
            return self._num_events // 2
        else:
            return self._num_spectral_lines

    @property
    def num_events(self) -> int:
        return self._num_events

    @property
    def window(self) -> Optional[str]:
        return self._window

    @property
    def hop_size(self) -> int:
        return self._hop_size

    @classmethod
    def build_op_definition(cls) -> pb.OperatorDef:
        return pb.OperatorDef(
            key="FFT",
            attributes=[
                pb.OperatorDef.Attribute(
                    key="num_events",
                    type=pb.OperatorDef.Attribute.Type.INTEGER_64,
                ),
                pb.OperatorDef.Attribute(
                    key="hop_size",
                    type=pb.OperatorDef.Attribute.Type.INTEGER_64,
                ),
                pb.OperatorDef.Attribute(
                    key="window",
                    type=pb.OperatorDef.Attribute.Type.STRING,
                    is_optional=True,
                ),
                pb.OperatorDef.Attribute(
                    key="num_spectral_lines",
                    type=pb.OperatorDef.Attribute.Type.INTEGER_64,
                    is_optional=True,
                ),
            ],
            inputs=[pb.OperatorDef.Input(key="input")],
            outputs=[pb.OperatorDef.Output(key="output")],
        )


operator_lib.register_operator(FastFourierTransform)


@compile
def fast_fourier_transform(
    input: EventSetOrNode,
    *,
    num_events: int,
    hop_size: Optional[int] = None,
    window: Optional[str] = None,
    num_spectral_lines: Optional[int] = None,
) -> EventSetOrNode:
    return FastFourierTransform(
        input=input,
        num_events=num_events,
        hop_size=hop_size if hop_size is not None else num_events // 2,
        window=window,
        num_spectral_lines=num_spectral_lines,
    ).outputs[
        "output"
    ]  # type: ignore
