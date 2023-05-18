workspace(name = "temporian")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_python",
    sha256 = "81cbfc16dd1c022c4761267fa8b2feb881aaea9c3e1143f2e64630a1ad18c347",
    strip_prefix = "rules_python-0.16.1",
    url = "https://github.com/bazelbuild/rules_python/archive/refs/tags/0.16.1.zip",
)

http_archive(
    name = "bazel_skylib",
    sha256 = "74d544d96f4a5bb630d465ca8bbcfe231e3594e5aae57e1edbf17a6eb3ca2506",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz",
    ],
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

http_archive(
    name = "pybind11_bazel",
    sha256 = "37aca5f00771645b840fa84c71f450c3f9b7f78e65a7cc88bf7bf883449ff802",
    strip_prefix = "pybind11_bazel-master",
    urls = ["https://github.com/pybind/pybind11_bazel/archive/refs/heads/master.zip"],
)

http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    sha256 = "2db4dc881c89680da512c7f3e76ea4306b97e16d29058c1513709c6b80f6b57e",
    strip_prefix = "pybind11-master",
    urls = ["https://github.com/pybind/pybind11/archive/refs/heads/master.zip"],
)

load("@pybind11_bazel//:python_configure.bzl", "python_configure")

python_configure(name = "local_config_python")

http_archive(
    name = "com_google_protobuf",
    sha256 = "04e1ed9664d1325b43723b6a62a4a41bf6b2b90ac72b5daee288365aad0ea47d",
    strip_prefix = "protobuf-3.20.3",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/v3.20.3.zip"],
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()
