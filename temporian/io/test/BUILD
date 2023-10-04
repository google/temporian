package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Tests
# =====
py_test(
    name = "pandas_test",
    srcs = ["pandas_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/numpy
        # already_there/pandas
        "//temporian/implementation/numpy/data:io",
        "//temporian/io:pandas",
        "//temporian/implementation/numpy/operators/test:utils",
    ],
)

py_test(
    name = "tensorflow_test",
    srcs = ["tensorflow_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/numpy
        # already_there/tensorflow
        "//temporian/implementation/numpy/data:io",
        "//temporian/io:tensorflow",
        "//temporian/implementation/numpy/operators/test:utils",
    ],
)
