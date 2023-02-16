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
        "//temporian/implementation/numpy/operators:sum",
        "//temporian/implementation/numpy/operators/tests/sum/test_data:correct_sum",
        "//temporian/implementation/numpy/operators/tests/sum/test_data:different_timestamps",
        "//temporian/implementation/numpy/operators/tests/sum/test_data:different_indexes",
        "//temporian/implementation/numpy/operators/tests/sum/test_data:sum_with_nan",
    ],
)
