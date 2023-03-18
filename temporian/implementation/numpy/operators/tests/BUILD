package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Tests
# =====
py_test(
    name = "arithmetic_test",
    srcs = ["arithmetic_test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        "//temporian/core/data:event",
        "//temporian/core/data:sampling",
        "//temporian/core/operators:arithmetic",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/operators:arithmetic",
    ],
)

py_test(
    name = "assign_test",
    srcs = ["assign_test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        "//temporian/core/data:event",
        "//temporian/core/data:sampling",
        "//temporian/core/operators:assign",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:sampling",
        "//temporian/implementation/numpy/operators:assign",
    ],
)

py_test(
    name = "calendar_day_of_month_test",
    srcs = ["calendar_day_of_month_test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        "//temporian/core/data:event",
        "//temporian/core/data:sampling",
        "//temporian/core/operators:calendar_day_of_month",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:sampling",
        "//temporian/implementation/numpy/operators:calendar_day_of_month",
    ],
)

py_test(
    name = "lag_test",
    srcs = ["lag_test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        "//temporian/core/operators:lag",
        "//temporian/implementation/numpy/operators:lag",
    ],
)

py_test(
    name = "select_test",
    srcs = ["select_test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/operators:select",
    ],
)

py_test(
    name = "simple_moving_average_test",
    srcs = ["simple_moving_average_test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        # pandas dep,
        "//temporian/implementation/numpy/operators:simple_moving_average",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/data:dtype",
        "//temporian/core/operators:simple_moving_average",
    ],
)

py_test(
    name = "propagate",
    srcs = ["propagate.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:dtype",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/operators:propagate",
        "//temporian/implementation/numpy/operators:propagate",
    ],
)
