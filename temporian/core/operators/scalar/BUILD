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
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/core/data/dtypes:dtype",
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
        "//temporian/core/data/dtypes:dtype",
    ],
)

py_library(
    name = "relational_scalar",
    srcs = ["relational_scalar.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/core/data/dtypes:dtype",
    ],
)
