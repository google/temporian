package(
    default_visibility=["//visibility:public"],
    licenses=["notice"],
)

py_test(
    name="select_test",
    srcs=["select_test.py"],
    srcs_version="PY3",
    deps=[
        "//temporian/implementation/numpy/data:io",
        "//temporian/beam/test:utils",
        "//temporian/core/operators:select",
    ],
)
