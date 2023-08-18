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
        ":display_utils",
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
        "//temporian/utils:typecheck",
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

py_library(
    name = "display_utils",
    srcs = ["display_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:dtype",
        "//temporian/core/data:duration_utils",
        "//temporian/utils:config",
    ],
)
