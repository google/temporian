package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "arithmetic",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":addition",
        ":base",
        ":division",
        ":floordiv",
        ":multiplication",
        ":subtraction",
    ],
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/operators:base",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "addition",
    srcs = ["addition.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:event",
    ],
)

py_library(
    name = "subtraction",
    srcs = ["subtraction.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:event",
    ],
)

py_library(
    name = "multiplication",
    srcs = ["multiplication.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:event",
    ],
)

py_library(
    name = "division",
    srcs = ["division.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:event",
    ],
)

py_library(
    name = "floordiv",
    srcs = ["floordiv.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:event",
    ],
)
