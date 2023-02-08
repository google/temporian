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
    name = "prototype_test",
    srcs = ["prototype_test.py"],
    data = ["test_data/prototype"],
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        # pandas dep,
        "//temporian/core:evaluator",
        "//temporian/core/data:event",
        "//temporian/core/data:sampling",
        "//temporian/core/operators:assign",
        "//temporian/core/operators:select",
        "//temporian/core/operators:simple_moving_average",
        "//temporian/implementation/pandas/data:event",
    ],
)

py_test(
    name = "prototype_test_numpy",
    srcs = ["prototype_test_numpy.py"],
    data = ["test_data/prototype"],
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        "//temporian/core:evaluator",
        "//temporian/core/data:event",
        "//temporian/core/data:sampling",
        "//temporian/core/operators:select",
        "//temporian/implementation/numpy/data:event",
    ],
)
