package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "operators",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":arithmetic",
        ":assign",
        ":base",
        ":calendar_day_of_month",
        ":lag",
        ":select",
        ":simple_moving_average",
    ],
)

py_library(
    name = "arithmetic",
    srcs = ["arithmetic.py"],
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
    name = "assign",
    srcs = ["assign.py"],
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
    name = "calendar_base",
    srcs = ["calendar_base.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/data:dtype",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "calendar_day_of_month",
    srcs = ["calendar_day_of_month.py"],
    srcs_version = "PY3",
    deps = [
        ":calendar_base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:event",
    ],
)

py_library(
    name = "calendar_day_of_week",
    srcs = ["calendar_day_of_week.py"],
    srcs_version = "PY3",
    deps = [
        ":calendar_base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:event",
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
    name = "simple_moving_average",
    srcs = ["simple_moving_average.py"],
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
