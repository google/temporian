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
"""
Basic benchmarking script for temporian.

The script creates two events, applies some operators, and evaluates the graph.
"""

import numpy as np
import pandas as pd
import temporian as tp
from temporian.implementation.numpy.data.event import NumpyEvent

# Make results reproducible
np.random.seed(0)

# Control size of benchmark , N = 300000 takes about 5s to run
N = 300000

STORE, PRODUCT, TIMESTAMP, SALES, COSTS = (
    "store_id",
    "product_id",
    "timestamp",
    "sales",
    "costs",
)


@profile
def main():
    print(f"Running lag benchmark with N={N}...")

    # Integer ids from 0 to 9
    ids = list(range(int(10)))

    timestamps = np.sort(np.random.randn(N) * 100)
    product_ids = np.random.choice(ids, N)
    store_ids = np.random.choice(ids, N)

    event_data = NumpyEvent.from_dataframe(
        pd.DataFrame(
            {
                STORE: store_ids,
                PRODUCT: product_ids,
                TIMESTAMP: timestamps,
                SALES: np.random.randn(N) * 100,
            }
        ),
        index_names=[STORE, PRODUCT],
    )

    event = event_data.schema()

    sma = tp.sma(event, window_length=10)

    res: NumpyEvent = tp.evaluate(
        sma,
        input_data={
            event: event_data,
        },
        check_execution=False,
    )

    # Print output's first row, useful to check reproducibility
    print(res._first_index_features)


if __name__ == "__main__":
    main()
