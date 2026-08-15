# Release process

DocChrono publishes through GitHub Actions and PyPI Trusted Publishing. No long-lived PyPI token
is stored in the repository.

1. Ensure CI passes on CPython 3.11 through 3.13.
2. Set `project.version` and `docchrono.__version__` to the same PEP 440 version.
3. Update `CHANGELOG.md`.
4. Create a signed `v<version>` tag and a draft GitHub release.
5. For a release candidate, mark the GitHub release as a prerelease. The protected `testpypi`
   environment publishes it to TestPyPI.
6. Install the candidate with `--index-url https://test.pypi.org/simple/ --no-deps`; obtain normal
   dependencies from PyPI separately.
7. For production, publish a non-prerelease GitHub release. The protected `pypi` environment must
   require final human approval.
8. Verify PyPI metadata and attestations, then install the exact version in a clean environment.

Configure pending Trusted Publishers for repository `docchrono/docchrono`, workflow
`release.yml`, and environments `testpypi` and `pypi`. Protect release tags and the workflow file.
PyPI filenames are immutable; a corrected artifact always requires a new version.
