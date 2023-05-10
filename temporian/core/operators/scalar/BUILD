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
        ":base",
        ":relational_scalar",
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
    name = "arithmetic_scalar",
    srcs = ["arithmetic_scalar.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:node",
    ],
)

py_library(
    name = "relational_scalar",
    srcs = ["relational_scalar.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:dtype",
        "//temporian/core/data:feature",
        "//temporian/core/data:node",
    ],
)