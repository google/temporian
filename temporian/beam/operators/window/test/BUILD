package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)


py_test(
    name="moving_sum_test",
    srcs=["moving_sum_test.py"],
    srcs_version="PY3",
    deps=[
        "//temporian/implementation/numpy/data:io",
        "//temporian/beam/test:utils",
        "//temporian/core/operators/window:moving_sum",
    ],
)
