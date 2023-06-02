# Building Temporian

Temporian is not a pure Python package, so special care is required when
packaging it for distribution. This document outlines how to build a Temporian
package that can be distributed on PyPi.

## Package configuration

Packages are built through Poetry. Configuration of the build options is through
`pyproject.toml`.

Unfortunately, building non-pure packages through Poetry is still somewhat
unstable. Notably, the build uses a custom build script located under
`config/build.py`. The definition of these custom build scripts is being
discussed in this
[Github issue](https://github.com/python-poetry/poetry/issues/2740) on the
Poetry repository.

If needed, the `build.py` can also serve as the basis for a handcrafted
`setup.py` for building through other packaging tools.

If new protos or C++-extensions are added to the project, they must be added
(manually) to `config/move_generated_files.sh`.

## Linux builds

For Linux builds to be compatible with as many distributions as possible, builds
within the manylinux2014 container are encouraged.

We use the TFX manylinux docker that includes a working Bazel installation. Run
the following command to start the docker (might require superuser permissions):

```sh
./tools/start_compile_docker.sh
```

Within the docker, run the following command

```sh
PYTHON_VERSION=<version> ./tools/build_manylinux_all.sh
```

where `<version>` is one of `38`, `39`, `310`, `311`.

Or, in case you want to build for all supported Python versions

```sh
./tools/build_manylinux_all.sh
```

This will place the manylinux packages in the `dist/` directory under
Temporian's root.

## MacOS builds

Simply activate the desired Python version (e.g. using Pyenv) install Poetry
and run

```sh
poetry build
```

Note that separate builds for ARM64 and Intel Macs are necessary

## Windows builds

TODO
