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
        ":feature",
        ":sampling",
        "//temporian/core/data:dtype",
        "//temporian/core/data:duration",
        "//temporian/core/data:event",
        "//temporian/core/data:sampling",
        "//temporian/utils:string",
    ],
)

py_library(
    name = "feature",
    srcs = ["feature.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:dtype",
        "//temporian/core/data:feature",
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

py_library(
    name = "plotter",
    srcs = [
        "plotter.py",
        "plotter_bokeh.py",
        "plotter_matplotlib.py",
    ],
    srcs_version = "PY3",
    deps = [":event"],
)
