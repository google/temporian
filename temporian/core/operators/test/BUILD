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
        "//temporian/core/data:dtype",
        "//temporian/core/data:feature",
        "//temporian/core/data:node",
        "//temporian/core/operators:propagate",
    ],
)
