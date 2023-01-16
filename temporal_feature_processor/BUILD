
package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Libraries
# =========

py_library(
    name = "temporal_feature_processor",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        "//temporal_feature_processor/core",
        "//temporal_feature_processor/core:evaluator",
        "//temporal_feature_processor/core:operator_lib",
        "//temporal_feature_processor/core:processor",
        "//temporal_feature_processor/core/data:dtype",
        "//temporal_feature_processor/core/data:event",
        "//temporal_feature_processor/core/data:feature",
        "//temporal_feature_processor/core/data:sampling",
        "//temporal_feature_processor/core/operators",
        "//temporal_feature_processor/core/operators:base",
    ],
)
