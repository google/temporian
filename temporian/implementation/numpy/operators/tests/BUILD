package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Tests
# =====
py_test(
    name = "arithmetic_test",
    srcs = ["arithmetic_test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        "//temporian/core/data:event",
        "//temporian/core/data:sampling",
        "//temporian/core/operators:arithmetic",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/operators:arithmetic",
    ],
)

py_test(
    name = "assign_test",
    srcs = ["assign_test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        "//temporian/core/data:event",
        "//temporian/core/data:sampling",
        "//temporian/core/operators:assign",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:sampling",
        "//temporian/implementation/numpy/operators:assign",
    ],
)
