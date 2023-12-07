package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Tests
# =========

py_test(
    name = "string_test",
    srcs = ["string_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/absl/testing:parameterized
        "//temporian/utils:string",
    ],
)

py_test(
    name = "typecheck_test",
    srcs = ["typecheck_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/absl/testing:parameterized
        "//temporian/utils:typecheck",
        "//temporian/core:compilation",
        "//temporian/implementation/numpy/data:io",
    ],
)
