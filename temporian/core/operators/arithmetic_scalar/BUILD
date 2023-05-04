package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "arithmetic_scalar",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":add_scalar",
        ":base",
        ":divide_scalar",
        ":equal_scalar",
        ":floordiv_scalar",
        ":multiply_scalar",
        ":subtract_scalar",
    ],
)

py_library(
    name = "base",
    srcs = ["base.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:feature",
        "//temporian/core/data:node",
        "//temporian/core/operators:base",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "add_scalar",
    srcs = ["add_scalar.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:node",
    ],
)

py_library(
    name = "equal_scalar",
    srcs = ["equal_scalar.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:dtype",
        "//temporian/core/data:feature",
        "//temporian/core/data:node",
    ],
)

py_library(
    name = "subtract_scalar",
    srcs = ["subtract_scalar.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:node",
    ],
)

py_library(
    name = "multiply_scalar",
    srcs = ["multiply_scalar.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:node",
    ],
)

py_library(
    name = "divide_scalar",
    srcs = ["divide_scalar.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:node",
    ],
)

py_library(
    name = "floordiv_scalar",
    srcs = ["floordiv_scalar.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:node",
    ],
)
