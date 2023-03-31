package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_binary(
    name = "basic",
    srcs = ["basic.py"],
    srcs_version = "PY3",
    deps = ["//temporian"],
)

sh_binary(
    name = "time",
    srcs = ["scalene.sh"],
    data = [":basic"],
)

sh_binary(
    name = "memory",
    srcs = ["mprof.sh"],
    data = [":basic"],
)
