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

The script creates a node, applies an sma to it, and runs the graph.
"""

import numpy as np
import pandas as pd
import temporian as tp
from temporian.implementation.numpy.data.event_set import EventSet


def run(input_node, input_data, output_node):
    tp.run(output_node, input={input_node: input_data}, check_execution=False)


def main():
    print("Main")

    # Make results reproducible
    np.random.seed(0)

    # Control size of benchmark
    number_timestamps = 1_000_000

    feature_values = list(range(int(10)))
    index_values = list(range(int(5)))
    timestamps = np.sort(np.random.randn(number_timestamps) * number_timestamps)

    # all features are int categorical from 0 to 10
    index_1 = np.random.choice(index_values, number_timestamps)
    index_2 = np.random.choice(index_values, number_timestamps)
    feature_1 = np.random.choice(feature_values, number_timestamps)
    feature_2 = np.random.choice(feature_values, number_timestamps)
    feature_3 = np.random.choice(feature_values, number_timestamps)
    feature_4 = np.random.choice(feature_values, number_timestamps)
    feature_5 = np.random.choice(feature_values, number_timestamps)
    feature_6 = np.random.choice(feature_values, number_timestamps)

    input_data = tp.from_pandas(
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "index_1": index_1,
                "index_2": index_2,
                "feature_1": feature_1,
                "feature_2": feature_2,
                "feature_3": feature_3,
                "feature_4": feature_4,
                "feature_5": feature_5,
                "feature_6": feature_6,
            }
        ),
        indexes=[
            "index_1",
            "index_2",
        ],
    )

    indexes = ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"]

    input_node = input_data.node()
    output_node = input_node.add_index(indexes)

    run(input_node, input_data, output_node)

    print("Done")


if __name__ == "__main__":
    main()
