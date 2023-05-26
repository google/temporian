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
        "//temporian/core/operators/binary",
        "//temporian/core/operators/calendar:day_of_month",
        "//temporian/core/operators/calendar:day_of_week",
        "//temporian/core/operators/calendar:day_of_year",
        "//temporian/core/operators/calendar:hour",
        "//temporian/core/operators/calendar:iso_week",
        "//temporian/core/operators/calendar:minute",
        "//temporian/core/operators/calendar:month",
        "//temporian/core/operators/calendar:second",
        "//temporian/core/operators/calendar:year",
        "//temporian/core/operators/scalar",
        "//temporian/core/operators/window:moving_count",
        "//temporian/core/operators/window:moving_max",
        "//temporian/core/operators/window:moving_min",
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
        "//temporian/core/data:node",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:node",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "cast",
    srcs = ["cast.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:schema",
        "//temporian/core/data:node",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "unary",
    srcs = ["unary.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:schema",
        "//temporian/core/data:node",
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
        "//temporian/core/data:schema",
        "//temporian/core/data:node",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "filter",
    srcs = ["filter.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:dtype",
        "//temporian/core/data:schema",
        "//temporian/core/data:node",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "rename",
    srcs = ["rename.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:schema",
        "//temporian/core/data:node",
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
        "//temporian/core/data:schema",
        "//temporian/core/data:node",
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
        "//temporian/core/data:schema",
        "//temporian/core/data:node",
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
        "//temporian/core/data:schema",
        "//temporian/core/data:node",
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
        "//temporian/core/data:schema",
        "//temporian/core/data:node",
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
        "//temporian/core/data:schema",
        "//temporian/core/data:node",
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
        "//temporian/core/data:schema",
        "//temporian/core/data:node",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "since_last",
    srcs = ["since_last.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        ":select",
        "//temporian/core:operator_lib",
        "//temporian/core/data:duration",
        "//temporian/core/data:schema",
        "//temporian/core/data:node",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "unique_timestamps",
    srcs = ["unique_timestamps.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:schema",
        "//temporian/core/data:node",
        "//temporian/proto:core_py_proto",
    ],
)
