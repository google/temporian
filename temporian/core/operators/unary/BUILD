package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "unary",
    srcs = ["unary.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core:operator_lib",
        "//temporian/core/data:node",
        "//temporian/core/data/dtypes:dtype",
        "//temporian/core/operators:base",
        "//temporian/proto:core_py_proto",
    ],
)

py_library(
    name = "api_symbols",
    srcs = ["api_symbols.py"],
    srcs_version = "PY3",
    deps = [
        ":unary",
    ],
)
