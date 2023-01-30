package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Tests
# =====
py_test(
    name = "assign_test",
    srcs = ["assign_test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        "//temporian/implementation/pandas/operators:assign",
        "//temporian/implementation/pandas/operators/tests/assign/test_data:different_index",
        "//temporian/implementation/pandas/operators/tests/assign/test_data:repeated_timestamps",
        "//temporian/implementation/pandas/operators/tests/assign/test_data:with_idx_more_timestamps",
        "//temporian/implementation/pandas/operators/tests/assign/test_data:with_idx_same_timestamps",
    ],
)
