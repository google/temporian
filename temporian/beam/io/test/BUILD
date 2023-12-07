package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Tests
# =====

py_test(
    name = "dict_test",
    srcs = ["dict_test.py"],
    data = ["//temporian/test/test_data"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/absl/testing:parameterized
        "//temporian/beam/io:dict",
        "//temporian/core/data:node",
        "//temporian/core/data:dtype",
        "//temporian/implementation/numpy/data:io",
        "//temporian/io:csv",
    ],
)

py_test(
    name = "tensorflow_test",
    srcs = ["tensorflow_test.py"],
    data = ["//temporian/test/test_data"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/absl/testing:parameterized
        "//temporian/beam/io:tensorflow",
        "//temporian/beam/io:dict",
        "//temporian/core/data:node",
        "//temporian/core/data:dtype",
        "//temporian/implementation/numpy/data:io",
        "//temporian/io:csv",
        "//temporian/beam/io:csv",
        "//temporian/io:tensorflow",
        "//temporian/test:utils",
    ],
)

py_test(
    name = "csv_test",
    srcs = ["csv_test.py"],
    data = ["//temporian/test/test_data"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/absl/testing:parameterized
        "//temporian/beam/io:csv",
        "//temporian/beam/io:dict",
        "//temporian/core/data:node",
        "//temporian/core/data:dtype",
        "//temporian/implementation/numpy/data:io",
        "//temporian/io:csv",
        "//temporian/test:utils",
    ],
)
