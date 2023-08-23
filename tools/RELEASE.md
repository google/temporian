# Release a new version to PyPI

The Python package is built and published to PyPI when a new release is created in GitHub.

To create a new release, follow these steps:

1. Update the version number in `pyproject.toml` and `temporian/__init__.py` to the new version number (e.g. `1.3.2`), commit, and open and merge a PR to `main`, titled `Release v1.3.2` in this case.

2. Edit the [changelog](../CHANGELOG.md) by moving the latest changes to the new version's section and clearing the latest changes one.

3. Create a new [GitHub release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository#creating-a-release) with the new version's name prepended with `v`, e.g. `v1.3.2`.

4. Tag that commit as the new stable version with `git tag stable -f`, and push it with `git push origin stable -f`.
   - This gives us a way to easily find the latest stable version of the code in the GitHub tree (used for example by the tutorial notebooks to not open an unreleased version of the notebooks).
