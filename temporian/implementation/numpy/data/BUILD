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
    srcs = [
        "display_utils.py",
        "event_set.py",
    ],
    srcs_version = "PY3",
    deps = [
        # already_there/numpy
        ":dtype_normalization",
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
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
        ":dtype_normalization",
        "//temporian/core:evaluation",
        "//temporian/core/data:schema",
        "//temporian/core/operators:add_index",
    ],
)

py_library(
    name = "plotter_base",
    srcs = ["plotter_base.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:duration_utils",
        # already_there/numpy
    ],
)

py_library(
    name = "plotter",
    srcs = ["plotter.py"],
    srcs_version = "PY3",
    deps = [
        ":plotter_base",
        ":plotter_matplotlib",
        ":plotter_bokeh",
        ":event_set",
        ":dtype_normalization",
        "//temporian/core/data:duration_utils",
        # already_there/numpy
    ],
)

py_library(
    name = "plotter_matplotlib",
    srcs = ["plotter_matplotlib.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/numpy
        # force/matplotlib
        ":plotter_base",
        "//temporian/core/data:duration_utils",
    ],
)

py_library(
    name = "plotter_bokeh",
    srcs = ["plotter_bokeh.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/numpy
        # force/bokeh
        ":plotter_base",
        "//temporian/core/data:duration_utils",
    ],
)

py_library(
    name = "dtype_normalization",
    srcs = ["dtype_normalization.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/numpy
        "//temporian/core/data:dtype",
    ],
)
