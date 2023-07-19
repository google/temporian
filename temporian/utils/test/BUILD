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
        "//temporian/utils:string",
    ],
)

py_test(
    name = "rtcheck_test",
    srcs = ["rtcheck_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian/utils:rtcheck",
        "//temporian/core:compilation",
        "//temporian/implementation/numpy/data:io",
    ],
)
