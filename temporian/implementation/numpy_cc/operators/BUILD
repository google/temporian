package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

pybind_extension(
    name = "sample",
    srcs = ["sample.cc"],
)

py_library(
    name = "sample",
    data = [":sample.so"],
)

pybind_extension(
    name = "window",
    srcs = ["window.cc"],
)

py_library(
    name = "window",
    data = [":window.so"],
)
