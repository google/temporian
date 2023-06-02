package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Tests
# =====

py_test(
    name = "registered_operators_test",
    srcs = ["registered_operators_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/implementation/numpy:implementation_lib",
        "//temporian/implementation/numpy/operators:all_operators",
    ],
)
