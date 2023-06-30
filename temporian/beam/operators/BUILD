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
        ":add_index",
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

py_library(
    name="add_index",
    srcs=["add_index.py"],
    srcs_version="PY3",
    deps=[
        "//temporian/utils:config",
        "//temporian/beam/operators:base",
        "//temporian/core/operators:add_index",
        "//temporian/implementation/numpy/operators:add_index",
        "//temporian/beam:io",
        "//temporian/implementation/numpy/operators:base",
    ],
)
