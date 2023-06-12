package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "data",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
)

py_library(
    name = "node",
    srcs = ["node.py"],
    srcs_version = "PY3",
    deps = [
        ":schema",
        "//temporian/core/data/dtypes:dtype",
        "//temporian/utils:string",
    ],
)

py_library(
    name = "schema",
    srcs = ["schema.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data/dtypes:dtype",
    ],
)

py_library(
    name = "duration",
    srcs = ["duration.py"],
    srcs_version = "PY3",
    deps = [
        ":duration_utils",
    ],
)

py_library(
    name = "duration_utils",
    srcs = ["duration_utils.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/numpy
    ],
)
