package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "serialization",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":serialize",
    ],
)

py_library(
    name = "serialize",
    srcs = ["serialize.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/google/protobuf
        "//temporian/core:graph",
        "//temporian/core:operator_lib",
        "//temporian/core/data/dtypes:dtype",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/core/operators:base",
        "//temporian/proto:core_py_proto",
    ],
)
