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
        # already_there/numpy
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/core/data:duration_utils",
        "//temporian/utils:string",
        "//temporian/utils:config",
        "//temporian/core:event_set_ops",
    ],
)

py_library(
    name = "io",
    srcs = ["io.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/numpy
        ":event_set",
        "//temporian/core:evaluation",
        "//temporian/core/data:schema",
        "//temporian/core/operators:add_index",
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
    deps = [
        ":event_set",
        # already_there/numpy
        # force/matplotlib
        # force/bokeh
        "//temporian/core/data:duration_utils",
    ],
)
