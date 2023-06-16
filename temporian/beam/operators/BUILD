package(
    default_visibility=["//visibility:public"],
    licenses=["notice"],
)

# Libraries
# =========

py_library(
    name="base",
    srcs=["base.py"],
    srcs_version="PY3",
    deps=["//temporian/utils:config"],
)

py_library(
    name="operators",
    srcs=["__init__.py"],
    srcs_version="PY3",
    deps=[
        "//temporian/beam/operators/window:moving_sum",
        ":select",
    ],
)


py_library(
    name="select",
    srcs=["select.py"],
    srcs_version="PY3",
    deps=[
        "//temporian/utils:config",
        "//temporian/beam/operators:base",
        "//temporian/core/operators:select",
        "//temporian/implementation/numpy/operators:select",
        "//temporian/beam:io",
        "//temporian/implementation/numpy/operators:base",
    ],
)
