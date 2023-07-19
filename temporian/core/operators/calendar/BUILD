package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "calendar",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":day_of_month",
        ":day_of_week",
        ":day_of_year",
        ":hour",
        ":iso_week",
        ":minute",
        ":month",
        ":second",
        ":year",
    ],
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/operators:base",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "day_of_month",
    srcs = ["day_of_month.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:compilation",
        "//temporian/core:operator_lib",
        "//temporian/core:typing",
        "//temporian/core/data:node",
    ],
)

py_library(
    name = "day_of_week",
    srcs = ["day_of_week.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:compilation",
        "//temporian/core:operator_lib",
        "//temporian/core:typing",
        "//temporian/core/data:node",
    ],
)

py_library(
    name = "day_of_year",
    srcs = ["day_of_year.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:compilation",
        "//temporian/core:operator_lib",
        "//temporian/core:typing",
        "//temporian/core/data:node",
    ],
)

py_library(
    name = "hour",
    srcs = ["hour.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:compilation",
        "//temporian/core:operator_lib",
        "//temporian/core:typing",
        "//temporian/core/data:node",
    ],
)

py_library(
    name = "iso_week",
    srcs = ["iso_week.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:compilation",
        "//temporian/core:operator_lib",
        "//temporian/core:typing",
        "//temporian/core/data:node",
    ],
)

py_library(
    name = "minute",
    srcs = ["minute.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:compilation",
        "//temporian/core:operator_lib",
        "//temporian/core:typing",
        "//temporian/core/data:node",
    ],
)

py_library(
    name = "month",
    srcs = ["month.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:compilation",
        "//temporian/core:operator_lib",
        "//temporian/core:typing",
        "//temporian/core/data:node",
    ],
)

py_library(
    name = "second",
    srcs = ["second.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:compilation",
        "//temporian/core:operator_lib",
        "//temporian/core:typing",
        "//temporian/core/data:node",
    ],
)

py_library(
    name = "year",
    srcs = ["year.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:compilation",
        "//temporian/core:operator_lib",
        "//temporian/core:typing",
        "//temporian/core/data:node",
    ],
)
