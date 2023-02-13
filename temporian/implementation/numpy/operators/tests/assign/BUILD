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
        "//temporian/implementation/numpy/operators:assign",
        "//temporian/implementation/numpy/operators/tests/assign/test_data:assigned_repeated_timestamps",
        "//temporian/implementation/numpy/operators/tests/assign/test_data:assignee_repeated_timestamps",
        "//temporian/implementation/numpy/operators/tests/assign/test_data:less_assigned_indexes",
        "//temporian/implementation/numpy/operators/tests/assign/test_data:complete_timestamps",
        "//temporian/implementation/numpy/operators/tests/assign/test_data:different_index",
        "//temporian/implementation/numpy/operators/tests/assign/test_data:with_idx_more_timestamps",
        "//temporian/implementation/numpy/operators/tests/assign/test_data:with_idx_same_timestamps",
    ],
)
