# Contributing

## Contributing code

```bash
mamba env create -f ci/environment.yml
mamba activate virtualizarr-tests
pre-commit install
# git checkout -b new-feature
python -m pip install -e . --no-deps
python -m pytest ./virtualizarr --run-network-tests --cov=./ --cov-report=xml --verbose
```

## Contributing documentation

### Build the documentation locally

```bash
mamba env create -f ci/doc.yml
mamba activate virtualizarr-docs
pip install -e . # From project's root - needed to generate API docs
cd docs # From project's root
rm -rf generated
make clean
make html
```

### Access the documentation locally

Open `docs/_build/html/index.html` in a web browser

## Making a release

1. Navigate to the [https://github.com/zarr-developers/virtualizarr/releases](https://github.com/zarr-developers/virtualizarr/releases) releases page.
2. Select draft a new release.
3. Select 'Choose a tag', then 'create a new tag'
4. Enter the name for the new tag following the [EffVer](https://jacobtomlinson.dev/effver/) versioning scheme (e.g., releasing v0.2.0 as the next release after v0.1.0 denotes that “some small effort may be required to make sure this version works for you”).
4. Click 'Generate Release Notes' to draft notes based on merged pull requests.
5. Edit the draft release notes for consistency.
6. Select 'Publish' to publish the release. This should automatically upload the new release to PyPI and Conda-Forge.
7. Create and merge a PR to add a new empty section to the `docs/releases.rst` for the next release in the future.
