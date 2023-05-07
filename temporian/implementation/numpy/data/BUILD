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
    name = "event_set",
    srcs = ["event_set.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:dtype",
        "//temporian/core/data:duration",
        "//temporian/core/data:node",
        "//temporian/core/data:sampling",
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
    deps = [":event_set"],
)
