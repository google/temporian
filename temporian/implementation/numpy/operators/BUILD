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
        ":cast",
        ":drop_index",
        ":filter",
        ":glue",
        ":lag",
        ":prefix",
        ":propagate",
        ":rename",
        ":sample",
        ":select",
        ":set_index",
        ":since_last",
        ":unary",
        ":unique_timestamps",
        "//temporian/implementation/numpy/operators/binary",
        "//temporian/implementation/numpy/operators/calendar:day_of_month",
        "//temporian/implementation/numpy/operators/calendar:day_of_week",
        "//temporian/implementation/numpy/operators/calendar:day_of_year",
        "//temporian/implementation/numpy/operators/calendar:hour",
        "//temporian/implementation/numpy/operators/calendar:iso_week",
        "//temporian/implementation/numpy/operators/calendar:minute",
        "//temporian/implementation/numpy/operators/calendar:month",
        "//temporian/implementation/numpy/operators/calendar:second",
        "//temporian/implementation/numpy/operators/calendar:year",
        "//temporian/implementation/numpy/operators/scalar",
        "//temporian/implementation/numpy/operators/window:moving_count",
        "//temporian/implementation/numpy/operators/window:moving_max",
        "//temporian/implementation/numpy/operators/window:moving_min",
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
    deps = ["//temporian/utils:config"],
)

py_library(
    name = "filter",
    srcs = ["filter.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators:filter",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event_set",
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
        "//temporian/implementation/numpy/data:event_set",
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
        "//temporian/implementation/numpy/data:event_set",
    ],
)

py_library(
    name = "drop_index",
    srcs = ["drop_index.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators:drop_index",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event_set",
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
        "//temporian/implementation/numpy/data:event_set",
    ],
)

py_library(
    name = "rename",
    srcs = ["rename.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators:rename",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event_set",
    ],
)

py_library(
    name = "set_index",
    srcs = ["set_index.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators:set_index",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event_set",
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
        "//temporian/implementation/numpy/data:event_set",
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
        "//temporian/implementation/numpy/data:event_set",
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
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy_cc/operators:operators_cc",
    ],
)

py_library(
    name = "since_last",
    srcs = ["since_last.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/data:duration",
        "//temporian/core/operators:since_last",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy_cc/operators:operators_cc",
    ],
)

py_library(
    name = "cast",
    srcs = ["cast.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators:cast",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event_set",
    ],
)

py_library(
    name = "unary",
    srcs = ["unary.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators:unary",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event_set",
    ],
)

py_library(
    name = "unique_timestamps",
    srcs = ["unique_timestamps.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/data:duration",
        "//temporian/core/operators:unique_timestamps",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy:utils",
        "//temporian/implementation/numpy/data:event_set",
    ],
)
