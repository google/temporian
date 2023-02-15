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

"""NumpySumOperator - correct sum

Tests correct functionality of the sum operator

"""
import numpy as np

from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling

sampling = NumpySampling(
    index=["store_id"],
    data={("A",): np.array([1, 1.5, 3, 3.5, 4, 10, 20])},
)

INPUT = NumpyEvent(
    data={
        ("A",): [
            NumpyFeature(
                name="sales",
                data=np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]),
            ),
        ],
    },
    sampling=sampling,
)


OUTPUT = NumpyEvent(
    data={
        ("A",): [
            NumpyFeature(
                name="lag_sales",
                data=np.array([np.nan, np.nan, 10.0, 11.0, 11.0, 14.0, 15.0]),
            ),
        ],
    },
    sampling=sampling,
)
