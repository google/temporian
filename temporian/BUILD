package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "temporian",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core:evaluation",
        "//temporian/core:graph",
        "//temporian/core:operator_lib",
        "//temporian/core:serialization",
        "//temporian/core/data:dtype",
        "//temporian/core/data:duration",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/core/operators",
        "//temporian/core/operators:base",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/data:io",
        "//temporian/implementation/numpy/data:plotter",
        "//temporian/io",
    ],
)
