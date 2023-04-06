package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_library(
    name = "scripts",
    srcs_version = "PY3",
    deps = [
        ":basic",
        ":sma",
    ],
)

py_binary(
    name = "basic",
    srcs = ["basic.py"],
    srcs_version = "PY3",
    deps = ["//temporian"],
)

py_binary(
    name = "sma",
    srcs = ["sma.py"],
    srcs_version = "PY3",
    deps = ["//temporian"],
)
