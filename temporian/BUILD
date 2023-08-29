package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "temporian",
    srcs = [
        "__init__.py",
        "help.py",
    ],
    srcs_version = "PY3",
    deps = [
        "//temporian/core:compilation",
        "//temporian/core:evaluation",
        "//temporian/core:serialization",
        "//temporian/core:typing",
        "//temporian/core/data:dtype",
        "//temporian/core/data:duration",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/core/operators:combine",
        "//temporian/core/operators:glue",
        "//temporian/core/operators:unary",
        "//temporian/core/operators/binary:arithmetic",
        "//temporian/core/operators/binary:logical",
        "//temporian/core/operators/binary:relational",
        "//temporian/core/operators/scalar:arithmetic_scalar",
        "//temporian/core/operators/scalar:relational_scalar",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/data:io",
        "//temporian/implementation/numpy/data:plotter",
        "//temporian/implementation/numpy/operators",
        "//temporian/io:csv",
        "//temporian/io:format",
        "//temporian/io:pandas",
        "//temporian/io:tensorflow",
        "//temporian/utils:config",
        "//temporian/utils:typecheck",
    ],
)
