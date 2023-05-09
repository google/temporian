package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "binary",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":arithmetic",
        ":base",
        ":logical",
        ":relational",
    ],
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/operators/binary",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators:base",
    ],
)

py_library(
    name = "arithmetic",
    srcs = ["arithmetic.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/binary",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event_set",
    ],
)

py_library(
    name = "relational",
    srcs = ["relational.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/binary",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event_set",
    ],
)

py_library(
    name = "logical",
    srcs = ["logical.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core/operators/binary",
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/data:event_set",
    ],
)
