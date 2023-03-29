package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "window",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:duration",
        "//temporian/core/operators/window:base",
        "//temporian/implementation/numpy/data:event",
    ],
)

py_library(
    name = "simple_moving_average",
    srcs = ["simple_moving_average.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/window:simple_moving_average",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event",
    ],
)

py_library(
    name = "moving_standard_deviation",
    srcs = ["moving_standard_deviation.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/window:moving_standard_deviation",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event",
    ],
)

py_library(
    name = "moving_sum",
    srcs = ["moving_sum.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/window:moving_sum",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event",
    ],
)
