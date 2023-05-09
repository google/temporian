package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension", "pybind_library")

pybind_library(
    name = "since_last",
    srcs = ["since_last.cc"],
    hdrs = ["since_last.h"],
)

pybind_library(
    name = "window",
    srcs = ["window.cc"],
    hdrs = ["window.h"],
)

pybind_library(
    name = "sample",
    srcs = ["sample.cc"],
    hdrs = ["sample.h"],
)

pybind_extension(
    name = "operators_cc",
    srcs = ["pyinit.cc"],
    deps = [
        ":sample",
        ":since_last",
        ":window",
    ],
)

py_library(
    name = "operators_cc",
    data = [":operators_cc.so"],
)
