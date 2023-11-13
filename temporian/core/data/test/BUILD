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
        "//temporian/core/data:dtype",
    ],
)
