package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Tests
# =====
py_test(
    name = "test_node",
    srcs = ["test_node.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/absl/testing:parameterized
        "//temporian/core/test:utils",
        "//temporian/implementation/numpy/data:event_set",
    ],
)

py_test(
    name = "test_dtype",
    srcs = ["test_dtype.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/absl/testing:parameterized
        "//temporian/core/data:dtype",
    ],
)


py_test(
    name = "schema_test",
    srcs = ["schema_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/absl/testing:parameterized
        "//temporian/core/data:schema",
        "//temporian/core:serialization",
    ],
)
