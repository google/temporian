package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "scalar",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":arithmetic_scalar",
        ":relational_scalar",
    ],
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/numpy
        "//temporian/core/operators/scalar:base",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators:base",
    ],
)

py_library(
    name = "arithmetic_scalar",
    srcs = ["arithmetic_scalar.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        # already_there/numpy
        "//temporian/core/operators/scalar",
        "//temporian/implementation/numpy:implementation_lib",
    ],
)

py_library(
    name = "relational_scalar",
    srcs = ["relational_scalar.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        # already_there/numpy
        "//temporian/core/operators/scalar",
        "//temporian/implementation/numpy:implementation_lib",
    ],
)
