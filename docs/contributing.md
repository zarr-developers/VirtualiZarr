# Contributing

## Contributing documentation

### Build the documentation locally

```bash
mamba env create -f ci/doc.yml
mamba activate docs
cd docs # From project's root
rm -rf generated
make clean
make html
```

### Access the documentation locally

Open `docs/_build/html/index.html` in a web browser

## Making a release

1. Navigate to the [https://github.com/zarr-developers/virtualizarr/releases](https://github.com/zarr-developers/virtualizarr/releases) release page.
2. Select draft a new release.
3. Select 'Choose a tag', then 'create a new tag'
4. Enter the name for the new tag following the [EffVer](https://jacobtomlinson.dev/effver/) versioning scheme (e.g., releasing v0.2.0 as the next release denotes that “some small effort may be required to make sure this version works for you”).
4. Click 'Generate Release Notes' to draft notes based on merged pull requests.
5. Edit the draft release notes for consistency.
6. Publish the release.
7. Navigate to the GitHub actions page and approve the deployment to run the PyPI release section in the `release` GitHub actions environment. This requires having privileges set in the [environment protection rules](https://github.com/zarr-developers/VirtualiZarr/settings/environments/3162544968/edit), which currently includes being part of the `virtualizarr-core-devs` GitHub team.
