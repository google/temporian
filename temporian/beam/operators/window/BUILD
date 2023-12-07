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

        "//temporian/beam:typing",
        "//temporian/beam/operators:base",
        "//temporian/core/operators/window:base",
        "//temporian/implementation/numpy/operators/window:base",
    ],
)

py_library(
    name = "moving_count",
    srcs = ["moving_count.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/window:moving_count",
        "//temporian/implementation/numpy/operators/window:moving_count",
        "//temporian/beam:implementation_lib",
        "//temporian/beam/operators:base",
        "//temporian/core/operators/window:base",
        "//temporian/beam:typing",
        # already_there/apache_beam
        "//temporian/implementation/numpy/operators/window:base",
    ],
)

py_library(
    name = "moving_max",
    srcs = ["moving_max.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/window:moving_max",
        "//temporian/implementation/numpy/operators/window:moving_max",
        "//temporian/beam:implementation_lib",
        "//temporian/implementation/numpy/operators/window:base",
    ],
)

py_library(
    name = "moving_min",
    srcs = ["moving_min.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/window:moving_min",
        "//temporian/implementation/numpy/operators/window:moving_min",
        "//temporian/beam:implementation_lib",
        "//temporian/implementation/numpy/operators/window:base",
    ],
)

py_library(
    name = "moving_standard_deviation",
    srcs = ["moving_standard_deviation.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/window:moving_standard_deviation",
        "//temporian/implementation/numpy/operators/window:moving_standard_deviation",
        "//temporian/beam:implementation_lib",
        "//temporian/implementation/numpy/operators/window:base",
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
        "//temporian/beam:implementation_lib",
        "//temporian/implementation/numpy/operators/window:base",
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
        "//temporian/beam:implementation_lib",
        "//temporian/implementation/numpy/operators/window:base",
    ],
)
