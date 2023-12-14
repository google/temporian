package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========
filegroup(
    name = "reference",
    srcs = glob(["src/reference/**/*.md"]),
)

py_test(
    name = "public_api_test",
    srcs = ["public_api_test.py"],
    data = ["//docs:reference"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/absl/testing:parameterized
        "//temporian",
    ],
)
