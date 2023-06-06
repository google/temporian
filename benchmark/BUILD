package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

sh_binary(
    name = "profile_time",
    srcs = ["profile_time.sh"],
    data = ["//benchmark/scripts"],
)

sh_binary(
    name = "profile_memory",
    srcs = ["profile_memory.sh"],
    data = ["//benchmark/scripts"],
)

py_binary(
    name = "benchmark_time",
    srcs = ["benchmark_time.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/numpy
        # already_there/pandas
        "//temporian",
    ],
)
