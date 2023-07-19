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
"""Basic profiling script for temporian.

The script creates two events, applies some operators, and runs the graph.
"""

import numpy as np
import pandas as pd
import temporian as tp
from temporian.implementation.numpy.data.event_set import EventSet

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
    print(f"Running basic benchmark with N={N}...")

    # Integer ids from 0 to 9
    ids = list(range(int(10)))

    timestamps = np.sort(np.random.randn(N) * 100)
    product_ids = np.random.choice(ids, N)
    store_ids = np.random.choice(ids, N)

    evset_1 = tp.from_pandas(
        pd.DataFrame(
            {
                STORE: store_ids,
                PRODUCT: product_ids,
                TIMESTAMP: timestamps,
                SALES: np.random.randn(N) * 100,
            }
        ),
        indexes=[STORE, PRODUCT],
    )

    evset_2 = tp.from_pandas(
        pd.DataFrame(
            {
                STORE: store_ids,
                PRODUCT: product_ids,
                TIMESTAMP: timestamps,
                COSTS: np.random.randn(N) * 100,
            }
        ),
        indexes=[STORE, PRODUCT],
    )

    node_1 = evset_1.node()
    node_2 = evset_2.node()
    node_2._sampling = node_1._sampling

    a = tp.glue(node_1, node_2)
    b = tp.prefix(tp.simple_moving_average(a, window_length=10.0), "sma_")
    c = tp.glue(a, tp.resample(b, a))

    res: EventSet = tp.run(
        c,
        input={
            node_1: evset_1,
            node_2: evset_2,
        },
        check_execution=False,
    )

    # Print output's first row, useful to check reproducibility
    print(res.first_index_data())


if __name__ == "__main__":
    main()
