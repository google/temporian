licenses(["notice"])  # Apache 2.0

sh_binary(
    name = "move_generated_files",
    srcs = ["move_generated_files.sh"],
    data = [
        "//temporian/implementation/numpy_cc/operators:operators_cc",
        "//temporian/proto:core_py_proto",
    ],
)
