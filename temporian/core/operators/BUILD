package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "api_symbols",
    srcs = ["api_symbols.py"],
    srcs_version = "PY3",
    deps = [
        ":add_index",
        ":begin",
        ":cast",
        ":drop_index",
        ":end",
        ":filter",
        ":glue",
        ":lag",
        ":leak",
        ":prefix",
        ":propagate",
        ":rename",
        ":resample",
        ":select",
        ":since_last",
        ":tick",
        ":unique_timestamps",
        "//temporian/core/operators/binary:api_symbols",
        "//temporian/core/operators/calendar:api_symbols",
        "//temporian/core/operators/scalar:api_symbols",
        "//temporian/core/operators/unary:api_symbols",
        "//temporian/core/operators/window:api_symbols",
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
        "//temporian/core/data:schema",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:node",
        "//temporian/core/data/dtypes:dtype",
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
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/core/data/dtypes:dtype",
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
        "//temporian/core/data:duration_utils",
        "//temporian/core/data:node",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "leak",
    srcs = ["leak.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:duration_utils",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
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
        "//temporian/core/data:node",
        "//temporian/core/data/dtypes:dtype",
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
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
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
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
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
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "add_index",
    srcs = ["add_index.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        ":drop_index",
        "//temporian/core:operator_lib",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "prefix",
    srcs = ["prefix.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
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
        ":resample",
        "//temporian/core:operator_lib",
        "//temporian/core/data:node",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "resample",
    srcs = ["resample.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
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
        "//temporian/core:operator_lib",
        "//temporian/core/data:node",
        "//temporian/core/data/dtypes:dtype",
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
        "//temporian/core/data:node",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "begin",
    srcs = ["begin.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "end",
    srcs = ["end.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "tick",
    srcs = ["tick.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:duration_utils",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/proto:core_py_proto",
    ],
)
