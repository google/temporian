package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_test(
    name = "code_examples_test",
    srcs = ["code_examples_test.py"],
    data = glob(["**/*.md"]),
    srcs_version = "PY3",
    deps = [
        "//temporian",
    ],
)

filegroup(
    name = "reference",
    srcs = glob(["src/reference/**/*.md"]),
)

py_test(
    name = "public_api_test",
    srcs = ["public_api_test.py"],
    data = [        "//docs:reference"    ],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        "//temporian",
    ],
)
