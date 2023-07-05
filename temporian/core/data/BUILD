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
        ":dtype",
        ":schema",
        "//temporian/core:mixins",
        "//temporian/utils:string",
    ],
)

py_library(
    name = "schema",
    srcs = ["schema.py"],
    srcs_version = "PY3",
    deps = [
        ":dtype",
    ],
)

py_library(
    name = "duration",
    srcs = ["duration.py"],
    srcs_version = "PY3",
    deps = [
    ],
)

py_library(
    name = "duration_utils",
    srcs = ["duration_utils.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/numpy
        ":duration",
    ],
)

py_library(
    name = "dtype",
    srcs = ["dtype.py"],
    srcs_version = "PY3",
)
