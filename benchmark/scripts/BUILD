package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_library(
    name = "scripts",
    srcs_version = "PY3",
    deps = [":basic"],
)

py_binary(
    name = "basic",
    srcs = ["basic.py"],
    srcs_version = "PY3",
    deps = ["//temporian"],
)
