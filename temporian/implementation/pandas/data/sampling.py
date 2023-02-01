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

"""We define a Sampling as a list of timestamps for each of the values of a given index.
For the time being, we can represent this as a pandas Index whose last level must be a DatetimeIndex.
"""

import pandas as pd


class PandasSampling(pd.MultiIndex):
    def __new__(
        cls,
        levels=None,
        codes=None,
        sortorder=None,
        names=None,
        dtype=None,
        copy=False,
        name=None,
        verify_integrity: bool = True,
    ) -> pd.MultiIndex:
        if not isinstance(levels[-1], pd.DatetimeIndex):
            raise ValueError(
                "The last level of a PandasSampling must be a DatetimeIndex."
            )
        return super().__new__(
            cls,
            levels,
            codes,
            sortorder,
            names,
            dtype,
            copy,
            name,
            verify_integrity,
        )
