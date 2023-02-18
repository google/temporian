package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Tests
# =====
py_test(
    name = "multiply_test",
    srcs = ["multiply_test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        "//temporian/core/data:event",
        "//temporian/core/data:sampling",
        "//temporian/core/operators:sum",
        "//temporian/core/operators:multiply",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:sampling",
        "//temporian/implementation/numpy/operators:multiply",
    ],
)
