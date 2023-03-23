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
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/data:sampling",
        "//temporian/core/operators:base",
        "//temporian/implementation/numpy/data:event",
    ],
)

# Tests
# =====

py_test(
    name = "operator_test",
    srcs = ["operator_test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        "//temporian/core/data:event",
        "//temporian/core/data:sampling",
        "//temporian/core/operators:base",
        "//temporian/proto:core_py_proto",
    ],
)

py_test(
    name = "processor_test",
    srcs = ["processor_test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/logging dep,
        # absl/testing:absltest dep,
        ":utils",
        "//temporian/core:processor",
        "//temporian/core/data:dtype",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/data:sampling",
        "//temporian/core/operators:base",
        "//temporian/proto:core_py_proto",
    ],
)

py_test(
    name = "serialize_test",
    srcs = ["serialize_test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/logging dep,
        # absl/testing:absltest dep,
        ":utils",
        "//temporian/core:processor",
        "//temporian/core:serialize",
        "//temporian/core/data:dtype",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/data:sampling",
        "//temporian/core/operators:base",
        "//temporian/proto:core_py_proto",
    ],
)

py_test(
    name = "evaluator_test",
    srcs = ["evaluator_test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/logging dep,
        # absl/testing:absltest dep,
        ":utils",
        "//temporian/core:evaluator",
        "//temporian/core/data:dtype",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/data:sampling",
        "//temporian/core/operators:base",
        "//temporian/proto:core_py_proto",
    ],
)

py_test(
    name = "registered_operators_test",
    srcs = ["registered_operators_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core:operator_lib",
        "//temporian/core/operators:all_operators",
    ],
)
