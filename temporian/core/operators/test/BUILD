package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Tests
# =====
py_test(
    name = "propagate_test",
    srcs = ["propagate_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/operators:propagate",
    ],
)

py_test(
    name = "join_test",
    srcs = ["join_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/operators:join",
    ],
)
