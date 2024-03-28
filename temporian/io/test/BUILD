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
        # already_there/absl/testing:parameterized
        # already_there/pandas
        "//temporian/test:utils",
        "//temporian/implementation/numpy/data:io",
        "//temporian/io:pandas",
    ],
)

py_test(
    name = "polars_test",
    srcs = ["polars_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/numpy
        # already_there/absl/testing:parameterized
        # already_there/polars
        "//temporian/test:utils",
        "//temporian/implementation/numpy/data:io",
        "//temporian/io:polars",
    ],
)

py_test(
    name = "numpy_test",
    srcs = ["numpy_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/numpy
        # already_there/absl/testing:parameterized
        "//temporian/test:utils",
        "//temporian/implementation/numpy/data:io",
        "//temporian/io:numpy",
    ],
)

py_test(
    name = "tensorflow_test",
    srcs = ["tensorflow_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/absl/testing:parameterized
        # already_there/numpy
        # already_there/tensorflow
        "//temporian/test:utils",
        "//temporian/implementation/numpy/data:io",
        "//temporian/io:tensorflow",
    ],
)

py_test(
    name = "parquet_test",
    srcs = ["parquet_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/numpy
        # already_there/pyarrow
        "//temporian/test:utils",
        "//temporian/implementation/numpy/data:io",
        "//temporian/io:parquet",
    ],
)