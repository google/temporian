package(
    default_visibility=["//visibility:public"],
    licenses=["notice"],
)

# Libraries
# =========

py_library(
    name="moving_sum",
    srcs=["moving_sum.py"],
    srcs_version="PY3",
    deps=[
        "//temporian/utils:config",
        "//temporian/beam/operators:base",
        "//temporian/core/operators/window:moving_sum",
        "//temporian/implementation/numpy/operators/window:moving_sum",
        "//temporian/beam:io",
        "//temporian/implementation/numpy/operators:base",
    ],
)
