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
    name = "dtype",
    srcs = ["dtype.py"],
    srcs_version = "PY3",
)

py_library(
    name = "node",
    srcs = ["node.py"],
    srcs_version = "PY3",
    deps = [
        ":feature",
        ":sampling",
        "//temporian/utils:string",
    ],
)

py_library(
    name = "sampling",
    srcs = ["sampling.py"],
    srcs_version = "PY3",
)

py_library(
    name = "feature",
    srcs = ["feature.py"],
    srcs_version = "PY3",
    deps = [
        ":dtype",
        ":sampling",
    ],
)

py_library(
    name = "duration",
    srcs = ["duration.py"],
    srcs_version = "PY3",
    deps = [],
)
