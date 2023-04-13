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
        ":equal",
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
    name = "equal",
    srcs = ["equal.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/boolean:equal_feature",
        "//temporian/core/operators/boolean:equal_scalar",
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
    name = "drop_index",
    srcs = ["drop_index.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators:drop_index",
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
    name = "set_index",
    srcs = ["set_index.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        ":utils",
        "//temporian/core/operators:set_index",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:sampling",
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

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/implementation/numpy/data:event",
    ],
)
