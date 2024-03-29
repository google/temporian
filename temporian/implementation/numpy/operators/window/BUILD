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
        # already_there/numpy
        "//temporian/core/operators/window:base",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators:base",
        "//temporian/core/data:duration_utils",
        "//temporian/implementation/numpy/data:dtype_normalization",
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
        "//temporian/implementation/numpy_cc/operators:operators_cc",
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
        "//temporian/implementation/numpy_cc/operators:operators_cc",
    ],
)

py_library(
    name = "moving_product",
    srcs = ["moving_product.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/window:moving_product",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy_cc/operators:operators_cc",
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
        "//temporian/implementation/numpy_cc/operators:operators_cc",
    ],
)

py_library(
    name = "moving_count",
    srcs = ["moving_count.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/numpy
        ":base",
        "//temporian/core/operators/window:moving_count",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy_cc/operators:operators_cc",
        "//temporian/core/data:duration_utils",
    ],
)

py_library(
    name = "moving_max",
    srcs = ["moving_max.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/window:moving_max",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy_cc/operators:operators_cc",
    ],
)

py_library(
    name = "moving_min",
    srcs = ["moving_min.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/window:moving_min",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy_cc/operators:operators_cc",
    ],
)
