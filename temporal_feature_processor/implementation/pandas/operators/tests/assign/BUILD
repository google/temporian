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
        "//temporal_feature_processor/implementation/pandas/operators/tests/assign/test_data",
        "//temporal_feature_processor/implementation/pandas/operators",
        # absl/testing:absltest dep,
    ],
)
