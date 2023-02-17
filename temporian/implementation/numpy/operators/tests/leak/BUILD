package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Tests
# =====
py_test(
    name = "leak_test",
    srcs = ["leak_test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        "//temporian/core/operators:leak",
        "//temporian/implementation/numpy/operators:lag",
    ],
)
