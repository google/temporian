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
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/operators/calendar:base",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:feature",
        "//temporian/implementation/numpy/operators:base",
    ],
)

py_library(
    name = "day_of_month",
    srcs = ["day_of_month.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/calendar:day_of_month",
        "//temporian/implementation/numpy:implementation_lib",
    ],
)

py_library(
    name = "day_of_week",
    srcs = ["day_of_week.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/calendar:day_of_week",
        "//temporian/implementation/numpy:implementation_lib",
    ],
)

py_library(
    name = "day_of_year",
    srcs = ["day_of_year.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/calendar:day_of_year",
        "//temporian/implementation/numpy:implementation_lib",
    ],
)

py_library(
    name = "hour",
    srcs = ["hour.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/calendar:hour",
        "//temporian/implementation/numpy:implementation_lib",
    ],
)

py_library(
    name = "iso_week",
    srcs = ["iso_week.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/calendar:iso_week",
        "//temporian/implementation/numpy:implementation_lib",
    ],
)

py_library(
    name = "minute",
    srcs = ["minute.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/calendar:minute",
        "//temporian/implementation/numpy:implementation_lib",
    ],
)

py_library(
    name = "month",
    srcs = ["month.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/calendar:month",
        "//temporian/implementation/numpy:implementation_lib",
    ],
)

py_library(
    name = "second",
    srcs = ["second.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/calendar:second",
        "//temporian/implementation/numpy:implementation_lib",
    ],
)

py_library(
    name = "year",
    srcs = ["year.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/calendar:year",
        "//temporian/implementation/numpy:implementation_lib",
    ],
)
