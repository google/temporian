package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Tests
# =====
py_test(
    name = "sum_test",
    srcs = ["sum_test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        "//temporian/implementation/pandas/operators:sum",
        "//temporian/implementation/pandas/operators/tests/sum/test_data:different_shape",
    ],
)
