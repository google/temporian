package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Tests
# =====
py_test(
    name = "lag_test",
    srcs = ["lag_test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        "//temporian/core/operators:lag",
        "//temporian/implementation/numpy/operators:lag",
        "//temporian/implementation/numpy/operators/tests/lag/test_data:correct_lag",
    ],
)
