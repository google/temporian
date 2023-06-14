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
