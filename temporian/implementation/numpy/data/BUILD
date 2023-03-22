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
    name = "event",
    srcs = ["event.py"],
    srcs_version = "PY3",
    deps = [
        # pandas dep,
        ":sampling",
        "//temporian/core/data:duration",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/data:sampling",
        "//temporian/core/data:dtype",
        "//temporian/utils:string",
    ],
)

py_library(
    name = "sampling",
    srcs = ["sampling.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/utils:string",
    ],
)
