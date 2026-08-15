from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

REQUIRED_PACKAGE_FILES = {
    "docchrono/__init__.py",
    "docchrono/case.py",
    "docchrono/cli.py",
    "docchrono/py.typed",
}


def _wheel_files(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        return set(archive.namelist())


def _sdist_files(path: Path) -> set[str]:
    with tarfile.open(path, mode="r:gz") as archive:
        names = archive.getnames()
    return {
        PurePosixPath(name).relative_to(PurePosixPath(name).parts[0]).as_posix()
        for name in names
        if len(PurePosixPath(name).parts) > 1
    }


def main() -> int:
    distribution_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "dist")
    wheels = tuple(distribution_dir.glob("docchrono-*.whl"))
    sdists = tuple(distribution_dir.glob("docchrono-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        print("expected exactly one DocChrono wheel and one source distribution", file=sys.stderr)
        return 1
    wheel_missing = REQUIRED_PACKAGE_FILES - _wheel_files(wheels[0])
    wheel_files = _wheel_files(wheels[0])
    required_license_names = {"LICENSE", "NOTICE", "THIRD_PARTY_LICENSES.md"}
    wheel_license_names = {
        PurePosixPath(name).name for name in wheel_files if ".dist-info/licenses/" in name
    }
    wheel_license_missing = required_license_names - wheel_license_names
    sdist_files = _sdist_files(sdists[0])
    sdist_required = {"README.md", "LICENSE", "NOTICE", "pyproject.toml", "src/docchrono/case.py"}
    sdist_missing = sdist_required - sdist_files
    if wheel_missing or wheel_license_missing or sdist_missing:
        print(f"wheel missing: {sorted(wheel_missing)}", file=sys.stderr)
        print(f"wheel licenses missing: {sorted(wheel_license_missing)}", file=sys.stderr)
        print(f"sdist missing: {sorted(sdist_missing)}", file=sys.stderr)
        return 1
    print(f"verified {wheels[0].name} and {sdists[0].name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
