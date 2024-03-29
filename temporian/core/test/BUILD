package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core:operator_lib",
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/operators:base",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/data:io",
        "//temporian/proto:core_py_proto",
    ],
)

# Tests
# =====

py_test(
    name = "magic_methods_test",
    srcs = ["magic_methods_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:parameterized
        # already_there/absl/testing:absltest
        ":utils",
        "//temporian/core/data:node",
        "//temporian/core/operators:base",
        "//temporian/core/operators:unary",
        "//temporian/core/operators/binary",
        "//temporian/core/operators/scalar",
        "//temporian/proto:core_py_proto",
        "//temporian/implementation/numpy/data:io",
        "//temporian/implementation/numpy/data:event_set",
    ],
)

py_test(
    name = "operator_test",
    srcs = ["operator_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/absl/testing:parameterized
        "//temporian/core/data:node",
        "//temporian/core/operators:base",
        "//temporian/proto:core_py_proto",
    ],
)

py_test(
    name = "graph_test",
    srcs = ["graph_test.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
        # already_there/absl/testing:absltest
        # already_there/absl/testing:parameterized
        "//temporian/core:graph",
    ],
)

py_test(
    name = "serialization_test",
    srcs = ["serialization_test.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
        # already_there/absl/logging
        # already_there/absl/testing:absltest
        # already_there/absl/testing:parameterized
        "//temporian",
        "//temporian/core:graph",
        "//temporian/core:serialization",
        "//temporian/core/data:dtype",
        "//temporian/implementation/numpy/data:io",
    ],
)

py_test(
    name = "evaluation_test",
    srcs = ["evaluation_test.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
        # already_there/absl/testing:absltest
        # already_there/absl/testing:parameterized
        "//temporian/core:evaluation",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian",
    ],
)

py_test(
    name = "registered_operators_test",
    srcs = ["registered_operators_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/absl/testing:parameterized
        "//temporian/core:operator_lib",
        "//temporian",
    ],
)

py_test(
    name = "compilation_test",
    srcs = ["compilation_test.py"],
    srcs_version = "PY3",
    deps = [
        # already_there/absl/testing:absltest
        # already_there/absl/testing:parameterized
        "//temporian/core:operator_lib",
        "//temporian",
    ],
)