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
        ":logical",
        ":relational",
    ],
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/numpy
        "//temporian/core/data:dtype",
        "//temporian/core/operators/binary:base",
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
        # already_there/numpy
        "//temporian/core/data:dtype",
        "//temporian/core/operators/binary",
        "//temporian/implementation/numpy:implementation_lib",
    ],
)

py_library(
    name = "relational",
    srcs = ["relational.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        # already_there/numpy
        "//temporian/core/data:dtype",
        "//temporian/core/operators/binary",
        "//temporian/implementation/numpy:implementation_lib",
    ],
)

py_library(
    name = "logical",
    srcs = ["logical.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        # already_there/numpy
        "//temporian/core/data:dtype",
        "//temporian/core/operators/binary",
        "//temporian/implementation/numpy:implementation_lib",
    ],
)
