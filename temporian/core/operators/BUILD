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
        ":glue",
        ":lag",
        ":prefix",
        ":propagate",
        ":sample",
        ":select",
        "//temporian/core/operators/arithmetic",
        "//temporian/core/operators/calendar:day_of_month",
        "//temporian/core/operators/calendar:day_of_week",
        "//temporian/core/operators/calendar:day_of_year",
        "//temporian/core/operators/calendar:hour",
        "//temporian/core/operators/calendar:iso_week",
        "//temporian/core/operators/calendar:minute",
        "//temporian/core/operators/calendar:month",
        "//temporian/core/operators/calendar:second",
        "//temporian/core/operators/calendar:year",
        "//temporian/core/operators/window:moving_count",
        "//temporian/core/operators/window:moving_standard_deviation",
        "//temporian/core/operators/window:moving_sum",
        "//temporian/core/operators/window:simple_moving_average",
    ],
)

py_library(
    name = "glue",
    srcs = ["glue.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:event",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "lag",
    srcs = ["lag.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:duration",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "drop_index",
    srcs = ["drop_index.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/data:sampling",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "select",
    srcs = ["select.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:duration",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "set_index",
    srcs = ["set_index.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/data:sampling",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "propagate",
    srcs = ["propagate.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        ":select",
        "//temporian/core:operator_lib",
        "//temporian/core/data:duration",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "sample",
    srcs = ["sample.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        ":select",
        "//temporian/core:operator_lib",
        "//temporian/core/data:duration",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "prefix",
    srcs = ["prefix.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        ":select",
        "//temporian/core:operator_lib",
        "//temporian/core/data:duration",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/proto:core_py_proto",
    ],
)
