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
        "//temporian/core/data:node",
        "//temporian/core/operators/binary",
        "//temporian/implementation/numpy:evaluation",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators/binary",
    ],
)

py_test(
    name = "relational_test",
    srcs = ["relational_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:node",
        "//temporian/core/operators/binary",
        "//temporian/implementation/numpy:evaluation",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators/binary",
    ],
)

py_test(
    name = "logical_test",
    srcs = ["logical_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:node",
        "//temporian/core/operators/binary",
        "//temporian/implementation/numpy:evaluation",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators/binary",
    ],
)

py_test(
    name = "arithmetic_scalar_test",
    srcs = ["arithmetic_scalar_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/operators/scalar",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators/scalar",
    ],
)

py_test(
    name = "relational_scalar_test",
    srcs = ["relational_scalar_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/operators/scalar",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators/scalar",
    ],
)

py_test(
    name = "arithmetic_multi_index_test",
    srcs = ["arithmetic_multi_index_test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        "//temporian/core/data:node",
        "//temporian/core/operators/binary",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators/binary",
        "//temporian/implementation/numpy:evaluation",
    ],
)

py_test(
    name = "cast_test",
    srcs = ["cast_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/operators:cast",
        "//temporian/implementation/numpy:evaluation",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators:cast",
    ],
)

py_test(
    name = "filter_test",
    srcs = ["filter_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/operators:filter",
        "//temporian/implementation/numpy:evaluation",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators:filter",
    ],
)

py_test(
    name = "glue_test",
    srcs = ["glue_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/operators:glue",
        "//temporian/implementation/numpy:evaluation",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators:glue",
    ],
)

py_test(
    name = "unary_test",
    srcs = ["unary_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/operators:unary",
        "//temporian/implementation/numpy:evaluation",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators:unary",
    ],
)

py_test(
    name = "calendar_day_of_month_test",
    srcs = ["calendar_day_of_month_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:node",
        "//temporian/core/operators/calendar:day_of_month",
        "//temporian/implementation/numpy:evaluation",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators/calendar:day_of_month",
    ],
)

py_test(
    name = "calendar_day_of_week_test",
    srcs = ["calendar_day_of_week_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:node",
        "//temporian/core/operators/calendar:day_of_week",
        "//temporian/implementation/numpy:evaluation",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators/calendar:day_of_week",
    ],
)

py_test(
    name = "calendar_day_of_year_test",
    srcs = ["calendar_day_of_year_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:node",
        "//temporian/core/operators/calendar:day_of_year",
        "//temporian/implementation/numpy:evaluation",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators/calendar:day_of_year",
    ],
)

py_test(
    name = "calendar_hour_test",
    srcs = ["calendar_hour_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:node",
        "//temporian/core/operators/calendar:hour",
        "//temporian/implementation/numpy:evaluation",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators/calendar:hour",
    ],
)

py_test(
    name = "calendar_iso_week_test",
    srcs = ["calendar_iso_week_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:node",
        "//temporian/core/operators/calendar:iso_week",
        "//temporian/implementation/numpy:evaluation",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators/calendar:iso_week",
    ],
)

py_test(
    name = "calendar_minute_test",
    srcs = ["calendar_minute_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:node",
        "//temporian/core/operators/calendar:minute",
        "//temporian/implementation/numpy:evaluation",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators/calendar:minute",
    ],
)

py_test(
    name = "calendar_month_test",
    srcs = ["calendar_month_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:node",
        "//temporian/core/operators/calendar:minute",
        "//temporian/implementation/numpy:evaluation",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators/calendar:month",
    ],
)

py_test(
    name = "calendar_second_test",
    srcs = ["calendar_second_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:node",
        "//temporian/core/operators/calendar:second",
        "//temporian/implementation/numpy:evaluation",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators/calendar:second",
    ],
)

py_test(
    name = "calendar_year_test",
    srcs = ["calendar_year_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:node",
        "//temporian/core/operators/calendar:year",
        "//temporian/implementation/numpy:evaluation",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators/calendar:year",
    ],
)

py_test(
    name = "lag_test",
    srcs = ["lag_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core:evaluation",
        "//temporian/core/operators:lag",
        "//temporian/implementation/numpy:evaluation",
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
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators:drop_index",
    ],
)

py_test(
    name = "select_test",
    srcs = ["select_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core:evaluation",
        "//temporian/core/operators:select",
        "//temporian/implementation/numpy:evaluation",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators:select",
    ],
)

py_test(
    name = "rename_test",
    srcs = ["rename_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core:evaluation",
        "//temporian/core/operators:rename",
        "//temporian/implementation/numpy:evaluation",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators:rename",
    ],
)

py_test(
    name = "add_index_test",
    srcs = ["add_index_test.py"],
    srcs_version = "PY3",
    deps = [
        # absl/testing:absltest dep,
        "//temporian/core/data:node",
        "//temporian/core/operators:add_index",
        "//temporian/implementation/numpy/data:event_set",
        "//temporian/implementation/numpy/operators:add_index",
    ],
)

py_test(
    name = "simple_moving_average_test",
    srcs = ["simple_moving_average_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/core/operators/window:simple_moving_average",
        "//temporian/implementation/numpy:evaluation",
        "//temporian/implementation/numpy/operators/window:simple_moving_average",
    ],
)

py_test(
    name = "moving_standard_deviation_test",
    srcs = ["moving_standard_deviation_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/core/operators/window:moving_standard_deviation",
        "//temporian/implementation/numpy/operators/window:moving_standard_deviation",
    ],
)

py_test(
    name = "moving_sum_test",
    srcs = ["moving_sum_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/core/operators/window:moving_sum",
        "//temporian/implementation/numpy/operators/window:moving_sum",
    ],
)

py_test(
    name = "moving_count_test",
    srcs = ["moving_count_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/core/operators/window:moving_count",
        "//temporian/implementation/numpy/operators/window:moving_count",
    ],
)

py_test(
    name = "moving_min_test",
    srcs = ["moving_min_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/core/operators/window:moving_min",
        "//temporian/implementation/numpy/operators/window:moving_min",
    ],
)

py_test(
    name = "moving_max_test",
    srcs = ["moving_max_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/core/operators/window:moving_max",
        "//temporian/implementation/numpy/operators/window:moving_max",
    ],
)

py_test(
    name = "propagate_test",
    srcs = ["propagate_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/core/operators:propagate",
        "//temporian/implementation/numpy:evaluation",
        "//temporian/implementation/numpy/operators:propagate",
    ],
)

py_test(
    name = "sample_test",
    srcs = ["sample_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/core/operators:resample",
        "//temporian/implementation/numpy:evaluation",
        "//temporian/implementation/numpy/operators:resample",
    ],
)

py_test(
    name = "prefix_test",
    srcs = ["prefix_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/core/operators:prefix",
        "//temporian/implementation/numpy:evaluation",
        "//temporian/implementation/numpy/operators:prefix",
    ],
)

py_test(
    name = "unique_timestamps_test",
    srcs = ["unique_timestamps_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/core/operators:unique_timestamps",
        "//temporian/implementation/numpy/operators:unique_timestamps",
    ],
)

py_test(
    name = "since_last_test",
    srcs = ["since_last_test.py"],
    srcs_version = "PY3",
    deps = [
        "//temporian/core/data:dtype",
        "//temporian/core/data:node",
        "//temporian/core/data:schema",
        "//temporian/core/operators:since_last",
        "//temporian/implementation/numpy/operators:since_last",
    ],
)
