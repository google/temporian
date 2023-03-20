package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "all_operators",
    srcs = ["all_operators.py"],
    srcs_version = "PY3",
    deps = [
        ":arithmetic",
        ":glue",
        ":lag",
        ":prefix",
        ":propagate",
        ":sample",
        ":select",
        "//temporian/implementation/numpy/operators/calendar:day_of_month",
        "//temporian/implementation/numpy/operators/calendar:day_of_week",
        "//temporian/implementation/numpy/operators/window:moving_count",
        "//temporian/implementation/numpy/operators/window:moving_standard_deviation",
        "//temporian/implementation/numpy/operators/window:moving_sum",
        "//temporian/implementation/numpy/operators/window:simple_moving_average",
    ],
)

py_library(
    name = "operators",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
)

py_library(
    name = "arithmetic",
    srcs = ["arithmetic.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators:arithmetic",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:feature",
    ],
)

py_library(
    name = "glue",
    srcs = ["glue.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators:glue",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:sampling",
    ],
)

py_library(
    name = "lag",
    srcs = ["lag.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/data:duration",
        "//temporian/core/operators:lag",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:feature",
        "//temporian/implementation/numpy/data:sampling",
    ],
)

py_library(
    name = "reset_index",
    srcs = ["reset_index.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/operators:reset_index",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:sampling",
    ],
)

py_library(
    name = "select",
    srcs = ["select.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators:select",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event",
    ],
)

py_library(
    name = "propagate",
    srcs = ["propagate.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/data:duration",
        "//temporian/core/operators:propagate",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy:utils",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:sampling",
    ],
)

py_library(
    name = "sample",
    srcs = ["sample.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/data:duration",
        "//temporian/core/operators:sample",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy:utils",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:sampling",
        "//temporian/implementation/numpy_cc/operators:sample",
    ],
)

py_library(
    name = "prefix",
    srcs = ["prefix.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/data:duration",
        "//temporian/core/operators:prefix",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy:utils",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:sampling",
    ],
)
