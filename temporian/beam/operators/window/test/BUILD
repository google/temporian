package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_test(
    name = "window_test",
    srcs = ["window_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/absl/testing:parameterized
        # already_there/google/protobuf:use_fast_cpp_protos
        "//temporian/implementation/numpy/data:io",
        "//temporian/beam/test:utils",
        "//temporian/core/operators/window:moving_sum",
    ],
)
