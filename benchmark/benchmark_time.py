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
Benchmark Python API.

Usage example:

# Run the full benchmark.
bazel run -c opt //benchmark:benchmark_time

# Only run the "add_index" runs.
bazel run -c opt //benchmark:benchmark_time -- -f=add_index

# Run add_index and from_pandas
bazel run -c opt //benchmark:benchmark_time -- --f add_index from_pandas

"""
import argparse
import time
from typing import List, NamedTuple, Union

import numpy as np
import pandas as pd
import temporian as tp


def _build_toy_dataset(
    n: int, data_prefix="", data2_is_categorical_integer=False
) -> tp.EventSet:
    """Builds a toy dataset with two features.

    Args:
        n: Number of timestamps.
        data_prefix: Optional prefix in the feature names.
        data2_is_categorical_integer: If true, the second feature is
            categorical. If false (default), the second feature is numerical.

    Returns:
        An EventSet containing the toy dataset.
    """

    np.random.seed(0)
    index_values = list(range(int(10)))
    timestamps = np.sort(np.random.randn(n) * n)
    index_1 = np.random.choice(index_values, n)
    index_2 = np.random.choice(index_values, n)
    data_1 = np.random.randn(n)
    if data2_is_categorical_integer:
        data_2 = np.random.choice(list(range(int(10))), n)
    else:
        data_2 = np.random.randn(n)

    return tp.from_pandas(
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "index_1": index_1,
                "index_2": index_2,
                data_prefix + "data_1": data_1,
                data_prefix + "data_2": data_2,
            }
        ),
        indexes=["index_1", "index_2"],
    )


def benchmark_simple_moving_average(runner):
    runner.add_separator()
    for n in [100, 10_000, 1_000_000]:
        ds = _build_toy_dataset(n)

        node = ds.node()
        output = tp.simple_moving_average(node, window_length=10.0)

        runner.benchmark(
            f"simple_moving_average:{n:_}",
            lambda: tp.run(output, input={node: ds}),
        )


def benchmark_select_and_glue(runner):
    runner.add_separator()
    for n in [100, 10_000, 1_000_000]:
        ds = _build_toy_dataset(n)

        node = ds.node()
        output = tp.glue(node["data_1"], node["data_2"])

        runner.benchmark(
            f"select_and_glue:{n:_}",
            lambda: tp.run(output, input={node: ds}),
        )


def benchmark_calendar_day_of_month(runner):
    runner.add_separator()
    for n in [100, 10_000, 1_000_000]:
        np.random.seed(0)
        timestamps = np.sort(np.random.randn(n) * 1_700_000_000).astype(
            "datetime64[s]"
        )
        ds = tp.from_pandas(pd.DataFrame({"timestamp": timestamps}))

        node = ds.node()
        output = tp.calendar_day_of_month(node)

        runner.benchmark(
            f"calendar_day_of_month:{n:_}",
            lambda: tp.run(output, input={node: ds}),
        )


def benchmark_sample(runner):
    runner.add_separator()
    for m in [100, 10_000, 1_000_000]:
        for n in [100, 10_000, 1_000_000]:
            ds_1 = _build_toy_dataset(m, "ds_1")
            ds_2 = _build_toy_dataset(n, "ds_2")

            node_1 = ds_1.node()
            node_2 = ds_2.node()
            output = tp.resample(node_1, node_2)

            runner.benchmark(
                f"sample:e{m:_}_s{n:_}",
                lambda: tp.run(output, input={node_1: ds_1, node_2: ds_2}),
            )


def benchmark_propagate(runner):
    runner.add_separator()
    for n in [100, 10_000, 1_000_000]:
        ds = _build_toy_dataset(n, data2_is_categorical_integer=True)
        node = ds.node()
        output = node["data_1"].propagate(node["data_2"])

        runner.benchmark(
            f"propagate:{n:_}",
            lambda: tp.run(output, input={node: ds}),
        )


def benchmark_cast(runner):
    runner.add_separator()
    for n in [100, 1_000_000]:
        for check in [False, True]:
            ds = _build_toy_dataset(n)

            node = ds.node()
            output = node.cast(
                {
                    "data_1": tp.int32,
                    "data_2": tp.float32,
                },
                check_overflow=check,
            )

            runner.benchmark(
                f"cast({check=}):{n}",
                lambda: tp.run(output, input={node: ds}),
            )


def benchmark_unique_timestamps(runner):
    runner.add_separator()
    for n in [100, 10_000, 1_000_000]:
        ds = _build_toy_dataset(n, data2_is_categorical_integer=True)
        node = ds.node()
        output = tp.unique_timestamps(node["data_1"])

        runner.benchmark(
            f"unique_timestamps:{n}",
            lambda: tp.run(output, input={node: ds}),
        )


def benchmark_from_pandas(runner):
    runner.add_separator()
    # TODO: Add num_timestamps = 100_000 and 1_000_000 when from_pandas is
    # fast enough.
    for num_timestamps in [10_000]:
        for num_indexes in [0, 1, 3, 5]:
            for num_index_values in [20]:
                for index_is_string in [False, True]:
                    np.random.seed(0)

                    dt = {"timestamp": np.sort(np.random.rand(num_timestamps))}
                    index_names = []

                    inde_values = list(range(num_index_values))
                    for index_idx in range(num_indexes):
                        index_name = f"index_{index_idx}"
                        index_values = np.random.choice(
                            inde_values, num_timestamps
                        )
                        if index_is_string:
                            index_values = [f"v_{x}" for x in index_values]
                        dt[index_name] = index_values
                        index_names.append(index_name)

                    dt["feature_int64"] = np.random.randint(
                        0, 1000, num_timestamps, np.int64
                    )
                    dt["feature_str"] = np.random.rand(num_timestamps)
                    df = pd.DataFrame(dt)

                    benchmark_name = (
                        f"from_pandas:s:{num_timestamps:_}_"
                        f"numidx:{num_indexes}_"
                        f"numidxval:{num_index_values}_"
                        f"idx:{'str' if index_is_string else 'int'}"
                    )

                    runner.benchmark(
                        benchmark_name,
                        lambda: tp.from_pandas(df, index_names),
                    )


def benchmark_add_index(runner):
    runner.add_separator()

    np.random.seed(0)
    for number_timestamps in [10_000, 100_000, 1_000_000]:
        feature_values = list(range(int(10)))
        index_values = list(range(int(5)))
        timestamps = np.sort(
            np.random.randn(number_timestamps) * number_timestamps
        )

        # all features are int categorical from 0 to 10
        index_1 = np.random.choice(index_values, number_timestamps)
        index_2 = np.random.choice(index_values, number_timestamps)
        feature_1 = np.random.choice(feature_values, number_timestamps)
        feature_2 = np.random.choice(feature_values, number_timestamps)
        feature_3 = np.random.choice(feature_values, number_timestamps)
        feature_4 = np.random.choice(feature_values, number_timestamps)
        feature_5 = np.random.choice(feature_values, number_timestamps)
        feature_6 = np.random.choice(feature_values, number_timestamps)

        evset = tp.from_pandas(
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
            indexes=["index_1", "index_2"],
        )

        node = evset.node()

        possible_indexes = [
            ["feature_1"],
            ["feature_1", "feature_2"],
            ["feature_1", "feature_2", "feature_3"],
            ["feature_1", "feature_2", "feature_3", "feature_4"],
            ["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
        ]

        for index in possible_indexes:
            output = node.add_index(index)
            runner.benchmark(
                f"add_index:s:{number_timestamps:_}:num_idx:{len(index)}",
                lambda: tp.run(output, input={node: evset}),
            )


class BenchmarkResult(NamedTuple):
    name: str
    wall_time_seconds: float
    cpu_time_seconds: float

    def __str__(self):
        return (
            f"{self.name:30}    "
            f"{self.wall_time_seconds:10.5f}    "
            f"{self.cpu_time_seconds:10.5f}"
        )


class Runner:
    def __init__(self):
        # Name and tunning time (in seconds) of each benchmarks.
        self._results: List[Union[BenchmarkResult, None]] = []

    def benchmark(self, name, callback, warmup_repetitions=1, repetitions=5):
        print(f"Running {name}")
        # Warmup
        for _ in range(warmup_repetitions):
            callback()

        begin_wall_time = time.perf_counter()
        begin_cpu_time = time.process_time()

        for _ in range(repetitions):
            callback()

        end_wall_time = time.perf_counter()
        end_cpu_time = time.process_time()

        result = BenchmarkResult(
            name=name,
            wall_time_seconds=(end_wall_time - begin_wall_time) / repetitions,
            cpu_time_seconds=(end_cpu_time - begin_cpu_time) / repetitions,
        )
        self._results.append(result)

    def add_separator(self):
        self._results.append(None)

    def print_results(self):
        sep_length = 64
        print("=" * sep_length)
        print(f"{'Name':30}    {'Wall time (s)':10}    {'CPU time (s)':10}")
        print("=" * sep_length)
        for idx, result in enumerate(self._results):
            if result is None:
                if idx != 0:
                    print("-" * sep_length, flush=True)
            else:
                print(result)
        print("=" * sep_length, flush=True)


def main():
    # parse benchmarks to run from command line
    parser = argparse.ArgumentParser(description="Execute a list of functions.")
    parser.add_argument(
        "-f",
        "--functions",
        nargs="*",
        help=(
            "List of function names to execute. If not provided, all functions"
            " will be executed."
        ),
    )
    args = parser.parse_args()

    print("Running benchmark")
    runner = Runner()

    benchmarks_to_run = [
        "from_pandas",
        "simple_moving_average",
        "select_and_glue",
        "calendar_day_of_month",
        "sample",
        "propagate",
        "cast",
        "unique_timestamps",
        "add_index",
    ]
    if args.functions is not None:
        benchmarks_to_run = args.functions

    for func_name in benchmarks_to_run:
        try:
            eval(f"benchmark_{func_name}")(runner)
        except NameError:
            print(f"Function '{func_name}' not found.")

    print("All results (again)")
    runner.print_results()


if __name__ == "__main__":
    main()
