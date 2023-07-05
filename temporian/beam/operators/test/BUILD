package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_test(
    name = "select_test",
    srcs = ["select_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/implementation/numpy/data:io",
        "//temporian/beam/test:utils",
        "//temporian/core/operators:select",
    ],
)

py_test(
    name = "add_index_test",
    srcs = ["add_index_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/implementation/numpy/data:io",
        "//temporian/beam/test:utils",
        "//temporian/core/operators:add_index",
    ],
)
