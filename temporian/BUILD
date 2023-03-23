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
        "//temporian/core:evaluator",
        "//temporian/core:operator_lib",
        "//temporian/core:processor",
        "//temporian/core:serialize",
        "//temporian/core/data:dtype",
        "//temporian/core/data:duration",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/data:sampling",
        "//temporian/core/operators:all_operators",
        "//temporian/core/operators:base",
        "//temporian/implementation/numpy/operators:all_operators",
        "//temporian/io:read_event",
        "//temporian/io:save_event",
    ],
)
