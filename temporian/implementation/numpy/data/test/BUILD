package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Tests
# =====

py_test(
    name = "event_set_test",
    srcs = ["event_set_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/implementation/numpy/data:io",
    ],
)

py_test(
    name = "plotter_test",
    srcs = ["plotter_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:parameterized
        # already_there/matplotlib
        "//temporian/implementation/numpy/data:io",
        "//temporian/implementation/numpy/data:plotter",
    ],
)

py_test(
    name = "io_test",
    srcs = ["io_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/numpy
        # already_there/pandas
        "//temporian/core/data/dtypes:dtype",
        "//temporian/core/data:schema",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/data:io",
    ],
)
