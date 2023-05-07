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
    deps = [
        ":base",
        ":moving_count",
        ":moving_standard_deviation",
        ":moving_sum",
        ":simple_moving_average",
    ],
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:dtype",
        "//temporian/core/data:feature",
        "//temporian/core/data:node",
        "//temporian/core/operators:base",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "simple_moving_average",
    srcs = ["simple_moving_average.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:dtype",
        "//temporian/core/data:duration",
        "//temporian/core/data:feature",
        "//temporian/core/data:node",
        "//temporian/core/data:sampling",
    ],
)

py_library(
    name = "moving_standard_deviation",
    srcs = ["moving_standard_deviation.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:dtype",
        "//temporian/core/data:duration",
        "//temporian/core/data:feature",
        "//temporian/core/data:node",
        "//temporian/core/data:sampling",
    ],
)

py_library(
    name = "moving_sum",
    srcs = ["moving_sum.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:duration",
        "//temporian/core/data:feature",
        "//temporian/core/data:node",
        "//temporian/core/data:sampling",
    ],
)

py_library(
    name = "moving_count",
    srcs = ["moving_count.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:dtype",
        "//temporian/core/data:duration",
        "//temporian/core/data:feature",
        "//temporian/core/data:node",
        "//temporian/core/data:sampling",
    ],
)

py_library(
    name = "moving_min",
    srcs = ["moving_min.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:dtype",
        "//temporian/core/data:duration",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/data:sampling",
    ],
)

py_library(
    name = "moving_max",
    srcs = ["moving_max.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:dtype",
        "//temporian/core/data:duration",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/data:sampling",
    ],
)
