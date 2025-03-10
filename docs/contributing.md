# Contributing

Contributions are welcome and encouraged! We ask only that all contributors follow the [Zarr Developers Code of Conduct](https://github.com/zarr-developers/.github/blob/main/CODE_OF_CONDUCT.md).

## Contributing code

Before opening a PR to contribute code you should check that your changes work by running the test suite locally.

```bash
mamba env create -f ci/environment.yml
mamba activate virtualizarr-tests
pre-commit install
# git checkout -b new-feature
python -m pip install -e . --no-deps
python -m pytest
```

You may also add the `--run-network-tests` option, which will run additional tests
that require downloading files over the network.  Skip this if you want the tests to run
faster or you have no internet access:

```bash
python -m pytest --run-network-tests
```

Further, the `pytest-cov` plugin is a test dependency, so you can generate a test
coverage report locally, if you wish (CI will automatically do so).  Here are some
examples:

```bash
python -m pytest --cov=.                     # Terminal (text) report (--cov=term)
python -m pytest --cov=. --cov=term-missing  # Terminal report showing missing coverage
python -m pytest --cov=. --cov=html          # HTML report written to htmlcov/index.html
```

To see all available `pytest` options added by the `pytest-cov` plugin, run
`python -m pytest -h`, or see the
[pytest-cov documentation](https://pytest-cov.readthedocs.io/en/latest/readme.html).

## Contributing documentation

Whilst the CI will build the updated documentation for each PR, it can also be useful to check that the documentation has rendered as expected by building it locally.

### Build the documentation locally

```bash
mamba env create -f ci/doc.yml
mamba activate virtualizarr-docs
python -m pip install -e .  # From project's root - needed to generate API docs
cd docs # From project's root
rm -rf generated
make clean
make html
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
11. Create and merge a PR to add a new empty section to the `docs/releases.rst` for the next release in the future. See [this commit](https://github.com/zarr-developers/VirtualiZarr/commit/e3912f08e22f2e3230af6eb1a2aacb5728822fa1) for an example (you can assume the next release will be numbered `vX.Y.Z+1`, but the number doesn't actually matter).
12. (Optional) Advertise the release on social media 📣
