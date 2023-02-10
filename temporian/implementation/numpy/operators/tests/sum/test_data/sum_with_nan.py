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

Tests that sum two values with np.nan result in np.nan

"""
import numpy as np

from temporian.implementation.numpy.data.event import NumpyEvent
from temporian.implementation.numpy.data.event import NumpyFeature
from temporian.implementation.numpy.data.sampling import NumpySampling

sampling_1 = NumpySampling(
    names=["store_id"],
    data={
        ("A",): np.array(
            ["2022-02-05", "2022-02-06", "2022-02-07"],
            dtype="datetime64",
        ),
        ("B",): np.array(["2022-02-05", "2022-02-06"], dtype="datetime64"),
        ("C",): np.array(["2022-02-05", "2022-02-06"], dtype="datetime64"),
    },
)

sampling_2 = sampling_1

INPUT_1 = NumpyEvent(
    data={
        ("A",): [
            NumpyFeature(
                name="sales",
                sampling=sampling_1,
                data=np.array([np.nan, 15, 16]),
            ),
        ],
        ("B",): [
            NumpyFeature(
                name="sales",
                sampling=sampling_1,
                data=np.array([np.nan, 11]),
            ),
        ],
        ("C",): [
            NumpyFeature(
                name="sales",
                sampling=sampling_1,
                data=np.array([9, np.nan]),
            ),
        ],
    },
    sampling=sampling_1,
)

INPUT_2 = NumpyEvent(
    data={
        ("A",): [
            NumpyFeature(
                name="costs",
                sampling=sampling_2,
                data=np.array([-14, -15, -16]),
            ),
        ],
        ("B",): [
            NumpyFeature(
                name="costs",
                sampling=sampling_2,
                data=np.array([np.nan, -11]),
            ),
        ],
        ("C",): [
            NumpyFeature(
                name="costs",
                sampling=sampling_2,
                data=np.array([-9, np.nan]),
            ),
        ],
    },
    sampling=sampling_2,
)

OUTPUT = NumpyEvent(
    data={
        ("A",): [
            NumpyFeature(
                name="sum_sales_costs",
                sampling=sampling_1,
                data=np.array([np.nan, 0, 0]),  # np.nan + value = np.nan
            ),
        ],
        ("B",): [
            NumpyFeature(
                name="sum_sales_costs",
                sampling=sampling_1,
                data=np.array([np.nan, 0]),  # np.nan + np.nan = np.nan
            ),
        ],
        ("C",): [
            NumpyFeature(
                name="sum_sales_costs",
                sampling=sampling_1,
                data=np.array([0, np.nan]),  # np.nan + np.nan = np.nan
            ),
        ],
    },
    sampling=sampling_1,
)
