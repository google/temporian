package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Tests
# =====

py_test(
    name = "api_test",
    srcs = ["api_test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/logging dep,
        # absl/testing:absltest dep,
        # pandas dep,
        "//temporian",
    ],
)

py_test(
    name = "prototype_test_numpy",
    srcs = ["prototype_test_numpy.py"],
    data = ["test_data/prototype"],
    srcs_version = "PY3",
    deps = [
        "//temporian",
        "//temporian/implementation/numpy/data:event",
    ],
)

py_test(
    name = "io_test",
    srcs = ["io_test.py"],
    data = ["test_data/io"],
    srcs_version = "PY3",
    deps = [
        # absl/logging dep,
        # absl/testing:absltest dep,
        # pandas dep,
        "//temporian",
        "//temporian/implementation/numpy/data:event",
    ],
)
