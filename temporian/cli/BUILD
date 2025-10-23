package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Binary
# =========

py_binary(
    name = "main",
    srcs = ["main.py"],
    deps=["//temporian",":view",],
)


# Library
# =========

py_library(
    name = "view",
    srcs = ["view.py"],
    deps=[
        "//temporian",
        # already_there/PIL:pil_with_tk
        ],
)

# Tests
# =========

py_test(
    name = "view_test",
    srcs = ["view_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/absl/testing:parameterized
        ":view",
    ],
)
