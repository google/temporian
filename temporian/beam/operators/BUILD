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
    ],
)
