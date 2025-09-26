# Contributing

Contributions are welcome and encouraged! We ask only that all contributors follow the [Zarr Developers Code of Conduct](https://github.com/zarr-developers/.github/blob/main/CODE_OF_CONDUCT.md).

## Contributing code

Before opening a PR to contribute code you should check that your changes work by running the test suite locally.

!!! important
    We use [pixi](https://pixi.sh/latest/) to manage dependencies, which you'll want to install to get started.

Run tests with the `pixi run --environment test run-tests` command. Some tests require downloading files over the network.
Use the `run-tests-no-network` task if you want to run tests faster or have no internet access:

```bash
# Run all tests
pixi run --environment test run-tests
# Skip tests that require a network connection
pixi run --environment test run-tests-no-network
```

You can also run tests in other environments:

```bash
pixi run --environment min-deps run-tests # Test with the minimal set of dependencies installed
pixi run --environment upstream run-tests # Test with unreleased versions of upstream libraries
# List which versions are installed in the `min-deps` environment
pixi list --environment min-deps
```

Further, the `pytest-cov` plugin is a test dependency, so you can generate a test
coverage report locally, if you wish (CI will automatically do so).  Here are some
examples:

```bash
pixi run --environment test run-tests-cov              # Terminal report showing missing coverage
pixi run --environment test run-tests-html-cov         # HTML report written to htmlcov/index.html
```

Rather than using pixi tasks (essentially aliases for running commands in a given shell), you can explicitly start
a shell within a given environment and execute `pytest` (or other commands) directly:

```bash
# Start a shell within the environment
pixi shell --environment test
# Run the tests
pytest virtualizarr
# Exit the shell
exit
```

If you run into issues with the development environment, here are some recommending steps:
- Update pixi using `pixi self-update` and then retry the development workflow.
- Clean up environments using `pixi clean` and then retry the development workflow.
- Manually find and clean the cache dir listed in `pixi info` and then retry the development workflow.
- Ask for help in the [VirtualiZarr channel of the Earthmover community slack](https://earthmover-community.slack.com/archives/C08EXCE8ZQX).

### Code standards

#### Pre-commit

All code must conform to the PEP8 standard. `VirtualiZarr` uses a set of `pre-commit` hooks and the `pre-commit` bot to format, type-check, and prettify the codebase. `pre-commit` can be installed locally by running:

```
python -m pip install pre-commit
```
The hooks can be installed locally by running:

```
pre-commit install
```

This would run the checks every time a commit is created locally. These checks will also run on every commit pushed to an open PR, resulting in some automatic styling fixes by the `pre-commit` bot. The checks will by default only run on the files modified by a commit, but the checks can be triggered for all the files by running:

```
pre-commit run --all-files
```

If you would like to skip the failing checks and push the code for further discussion, use the `--no-verify` option with `git commit`.

#### Private functions

`VirtualiZarr` uses the following convention for private functions:

- Functions are preceded with an `_` (single underscore) if they should only be used within that module and may change at any time
- Functions without a preceding `_` (single underscore) are treated as relatively stable by the rest of the codebase, but not for public use (i.e. they are stable developer API).
- Public functions are documented in the fully public API and should follow the backwards-compatibility expectations of Effective Effort Versioning.

## Contributing documentation

Whilst the CI will build the updated documentation for each PR, it can also be useful to check that the documentation has rendered as expected by building it locally.

### Build the documentation locally

```bash
pixi install --environment docs
pixi run build-docs
```
Pixi can also be used to serve continuously updating version of the documentation during development at [http://0.0.0.0:8000/](http://0.0.0.0:8000/).
This can be done by navigating to [http://0.0.0.0:8000/](http://0.0.0.0:8000/) in your browser after running:

```bash
pixi run serve-docs
```

### Access the documentation locally

Open `docs/_build/html/index.html` in a web browser (on MacOS you can do this from the terminal using `open docs/_build/html/index.html`).

## Making a release

Anyone with commit privileges to the repository can issue a release, and you should feel free to issue a release at any point in time when all the CI tests on `main` are passing.

1. Decide on the release version number for the new release, following the [EffVer](https://jacobtomlinson.dev/effver/) versioning scheme (e.g., releasing v0.2.0 as the next release after v0.1.0 denotes that “some small effort may be required to make sure this version works for you”).
2. Write a high-level summary of the changes in this release, and write it into the release notes in `docs/releases.rst`. Create and merge a PR which adds the summary and also changes the release notes to say today's date and the version number of the new release. Don't add the blank template for future releases yet.
3. Navigate to the [https://github.com/zarr-developers/virtualizarr/releases](https://github.com/zarr-developers/virtualizarr/releases) releases page.
4. Select 'Draft a new release'.
5. Select 'Choose a tag', then 'Create a new tag'
6. Enter the name for the new tag (i.e. the release version number).
7. Click 'Generate Release Notes' to draft notes based on merged pull requests, and paste the same release summary you wrote earlier at the top.
8. Edit the draft release notes for consistency.
9. Select 'Publish' to publish the release. This should automatically upload the new release to [PyPI](https://pypi.org/project/virtualizarr/) and [conda-forge](https://anaconda.org/conda-forge/virtualizarr).
10. Check that this has run successfully (PyPI should show the new version number very quickly, but conda-forge might take several hours).
11. Create and merge a PR to add a new empty section to the `docs/releases.rst` for the next release in the future. You can assume the next release will be numbered `vX.Y.Z+1` where `vX.Y.Z` is the release just issued, but the number doesn't actually matter at this point. Just copy this template:
    ```
    ## vX.Y.Z+1 (unreleased)

    ### New Features

    ### Breaking changes

    ### Bug fixes

    ### Documentation

    ### Internal changes
    ```
12. (Optional) Advertise the release on social media 📣
