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
        ":api_symbols",
        ":arithmetic",
        ":logical",
        ":relational",
    ],
)

py_library(
    name = "api_symbols",
    srcs = ["api_symbols.py"],
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
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/core/data/dtypes:dtype",
        "//temporian/core/operators:base",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "arithmetic",
    srcs = ["arithmetic.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:node",
        "//temporian/core/data/dtypes:dtype",
    ],
)

py_library(
    name = "logical",
    srcs = ["logical.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/core/data/dtypes:dtype",
    ],
)

py_library(
    name = "relational",
    srcs = ["relational.py"],
    srcs_version = "PY3",
    deps = [
        ":base",
        "//temporian/core:operator_lib",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/core/data/dtypes:dtype",
    ],
)
