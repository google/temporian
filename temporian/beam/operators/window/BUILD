package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/apache_beam
        "//temporian/beam:implementation_lib",
        "//temporian/beam/operators:base",
        "//temporian/beam/io:dict",
        "//temporian/implementation/numpy/operators:base",
    ],
)

py_library(
    name = "moving_sum",
    srcs = ["moving_sum.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/window:moving_sum",
        "//temporian/implementation/numpy/operators/window:moving_sum",
    ],
)

py_library(
    name = "simple_moving_average",
    srcs = ["simple_moving_average.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/window:simple_moving_average",
        "//temporian/implementation/numpy/operators/window:simple_moving_average",
    ],
)
