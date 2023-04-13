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
        "//temporian/core/data:event",
        "//temporian/core/data:sampling",
        "//temporian/core/operators:arithmetic",
        "//temporian/implementation/numpy:evaluator",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/operators:arithmetic",
    ],
)

py_test(
    name = "equal_test",
    srcs = ["equal_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:event",
        "//temporian/core/data:sampling",
        "//temporian/core/operators:select",
        "//temporian/core/operators/boolean:equal_scalar",
        "//temporian/implementation/numpy:evaluator",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/operators/boolean:equal",
    ],
)

py_test(
    name = "glue_test",
    srcs = ["glue_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:sampling",
        "//temporian/core/operators:glue",
        "//temporian/implementation/numpy:evaluator",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/operators:glue",
    ],
)

py_test(
    name = "calendar_day_of_month_test",
    srcs = ["calendar_day_of_month_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:event",
        "//temporian/core/data:sampling",
        "//temporian/core/operators/calendar:day_of_month",
        "//temporian/implementation/numpy:evaluator",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:feature",
        "//temporian/implementation/numpy/data:sampling",
        "//temporian/implementation/numpy/operators/calendar:day_of_month",
    ],
)

py_test(
    name = "calendar_day_of_week_test",
    srcs = ["calendar_day_of_week_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:event",
        "//temporian/core/data:sampling",
        "//temporian/core/operators/calendar:day_of_week",
        "//temporian/implementation/numpy:evaluator",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:feature",
        "//temporian/implementation/numpy/data:sampling",
        "//temporian/implementation/numpy/operators/calendar:day_of_week",
    ],
)

py_test(
    name = "calendar_day_of_year_test",
    srcs = ["calendar_day_of_year_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:event",
        "//temporian/core/data:sampling",
        "//temporian/core/operators/calendar:day_of_year",
        "//temporian/implementation/numpy:evaluator",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:feature",
        "//temporian/implementation/numpy/data:sampling",
        "//temporian/implementation/numpy/operators/calendar:day_of_year",
    ],
)

py_test(
    name = "calendar_hour_test",
    srcs = ["calendar_hour_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:event",
        "//temporian/core/data:sampling",
        "//temporian/core/operators/calendar:hour",
        "//temporian/implementation/numpy:evaluator",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:feature",
        "//temporian/implementation/numpy/data:sampling",
        "//temporian/implementation/numpy/operators/calendar:hour",
    ],
)

py_test(
    name = "calendar_iso_week_test",
    srcs = ["calendar_iso_week_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:event",
        "//temporian/core/data:sampling",
        "//temporian/core/operators/calendar:iso_week",
        "//temporian/implementation/numpy:evaluator",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:feature",
        "//temporian/implementation/numpy/data:sampling",
        "//temporian/implementation/numpy/operators/calendar:iso_week",
    ],
)

py_test(
    name = "calendar_minute_test",
    srcs = ["calendar_minute_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:event",
        "//temporian/core/data:sampling",
        "//temporian/core/operators/calendar:minute",
        "//temporian/implementation/numpy:evaluator",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:feature",
        "//temporian/implementation/numpy/data:sampling",
        "//temporian/implementation/numpy/operators/calendar:minute",
    ],
)

py_test(
    name = "calendar_second_test",
    srcs = ["calendar_second_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:event",
        "//temporian/core/data:sampling",
        "//temporian/core/operators/calendar:second",
        "//temporian/implementation/numpy:evaluator",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:feature",
        "//temporian/implementation/numpy/data:sampling",
        "//temporian/implementation/numpy/operators/calendar:second",
    ],
)

py_test(
    name = "calendar_year_test",
    srcs = ["calendar_year_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:event",
        "//temporian/core/data:sampling",
        "//temporian/core/operators/calendar:year",
        "//temporian/implementation/numpy:evaluator",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:feature",
        "//temporian/implementation/numpy/data:sampling",
        "//temporian/implementation/numpy/operators/calendar:year",
    ],
)

py_test(
    name = "lag_test",
    srcs = ["lag_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core:evaluator",
        "//temporian/core/operators:lag",
        "//temporian/implementation/numpy:evaluator",
        "//temporian/implementation/numpy/operators:lag",
    ],
)

py_test(
    name = "drop_index_test",
    srcs = ["drop_index_test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        "//temporian/core/operators:drop_index",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:sampling",
        "//temporian/implementation/numpy/operators:drop_index",
    ],
)

py_test(
    name = "select_test",
    srcs = ["select_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core:evaluator",
        "//temporian/core/operators:select",
        "//temporian/implementation/numpy:evaluator",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/operators:select",
    ],
)

py_test(
    name = "set_index_test",
    srcs = ["set_index_test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        "//temporian/core/data:event",
        "//temporian/core/data:sampling",
        "//temporian/core/operators:set_index",
        "//temporian/implementation/numpy/data:event",
        "//temporian/implementation/numpy/data:sampling",
        "//temporian/implementation/numpy/operators:set_index",
    ],
)

py_test(
    name = "simple_moving_average_test",
    srcs = ["simple_moving_average_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:dtype",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/operators/window:simple_moving_average",
        "//temporian/implementation/numpy:evaluator",
        "//temporian/implementation/numpy/operators/window:simple_moving_average",
        "//temporian/implementation/numpy_cc/operators:window",
    ],
)

py_test(
    name = "moving_standard_deviation_test",
    srcs = ["moving_standard_deviation_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:dtype",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/operators/window:moving_standard_deviation",
        "//temporian/implementation/numpy/operators/window:moving_standard_deviation",
        "//temporian/implementation/numpy_cc/operators:window",
    ],
)

py_test(
    name = "moving_sum_test",
    srcs = ["moving_sum_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:dtype",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/operators/window:moving_sum",
        "//temporian/implementation/numpy/operators/window:moving_sum",
        "//temporian/implementation/numpy_cc/operators:window",
    ],
)

py_test(
    name = "moving_count_test",
    srcs = ["moving_count_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:dtype",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/operators/window:moving_count",
        "//temporian/implementation/numpy/operators/window:moving_count",
        "//temporian/implementation/numpy_cc/operators:window",
    ],
)

py_test(
    name = "propagate_test",
    srcs = ["propagate_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:dtype",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/operators:propagate",
        "//temporian/implementation/numpy:evaluator",
        "//temporian/implementation/numpy/operators:propagate",
    ],
)

py_test(
    name = "sample_test",
    srcs = ["sample_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:dtype",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/operators:sample",
        "//temporian/implementation/numpy:evaluator",
        "//temporian/implementation/numpy/operators:sample",
    ],
)

py_test(
    name = "prefix_test",
    srcs = ["prefix_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:dtype",
        "//temporian/core/data:event",
        "//temporian/core/data:feature",
        "//temporian/core/operators:prefix",
        "//temporian/implementation/numpy:evaluator",
        "//temporian/implementation/numpy/operators:prefix",
    ],
)
