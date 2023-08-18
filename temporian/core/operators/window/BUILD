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
        ":moving_count",
        ":moving_max",
        ":moving_min",
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
        "//temporian/core/data:duration_utils",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
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
        "//temporian/core:compilation",
        "//temporian/core:operator_lib",
        "//temporian/core:typing",
        "//temporian/core/data:dtype",
        "//temporian/core/data:duration_utils",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/utils:typecheck",
    ],
)

py_library(
    name = "moving_standard_deviation",
    srcs = ["moving_standard_deviation.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:compilation",
        "//temporian/core:operator_lib",
        "//temporian/core:typing",
        "//temporian/core/data:dtype",
        "//temporian/core/data:duration_utils",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
    ],
)

py_library(
    name = "moving_sum",
    srcs = ["moving_sum.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        # already_there/numpy
        "//temporian/core:operator_lib",
        "//temporian/core/data:dtype",
        "//temporian/core:typing",
        "//temporian/core/data:duration_utils",
        "//temporian/core:compilation",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
    ],
)

py_library(
    name = "moving_count",
    srcs = ["moving_count.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:compilation",
        "//temporian/core:operator_lib",
        "//temporian/core:typing",
        "//temporian/core/data:dtype",
        "//temporian/core/data:duration_utils",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
    ],
)

py_library(
    name = "moving_min",
    srcs = ["moving_min.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:compilation",
        "//temporian/core:operator_lib",
        "//temporian/core:typing",
        "//temporian/core/data:dtype",
        "//temporian/core/data:duration_utils",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
    ],
)

py_library(
    name = "moving_max",
    srcs = ["moving_max.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:compilation",
        "//temporian/core:operator_lib",
        "//temporian/core:typing",
        "//temporian/core/data:dtype",
        "//temporian/core/data:duration_utils",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
    ],
)
