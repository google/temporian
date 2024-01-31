# Release a new version to PyPI

The Python package is built and published to PyPI when a new release is created in GitHub.

To create a new release, follow these steps:

1. Open a new branch with the version's name, e.g. `v1.3.2`.

2. Update the version number in `pyproject.toml`, `config/setup.py`, `tools/build.bat`, and `temporian/__init__.py` to the new version number (`1.3.2` in this case).

3. Edit the [changelog](../CHANGELOG.md) by moving the latest changes to the new version's section and clearing the latest changes one.

4. Check the PRs merged since the last release and add any missing important changes to the changelog.

5. Commit your changes and open and merge a PR to `main`, titled `Release v1.3.2` in this case.

6. Wait for the testing actions to pass on the merge commit.

7. Create a new [GitHub release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository#creating-a-release) with the new version's name (`v1.3.2` in this case) as both the tag and the release name. Click on `Generate release notes`, and add this version's changelog on top with the heading "## Release notes".

8. Publish the Release. This will trigger the GitHub Action that builds and publishes the package to PyPI, and will point the /stable docs to this new version.

9. Pull `main`, tag the latest commit as the new stable version with `git tag last-release -f`, and push it with `git push origin last-release -f`. This gives us a way to easily find the latest stable version of the code in the GitHub tree (used for example by the tutorial notebooks to not open an unreleased version of the notebooks).
