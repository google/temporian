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
        "//temporian/core/data:dtype",
        "//temporian/core/data:duration",
        "//temporian/core/data:event",
        "//temporian/core/data:sampling",
        "//temporian/utils:string",
    ],
)

py_library(
    name = "plotter",
    srcs = ["plotter.py"],
    srcs_version = "PY3",
    deps = [":event"],
)
