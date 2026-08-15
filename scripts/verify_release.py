from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: verify_release.py v<version>", file=sys.stderr)
        return 2
    tag = sys.argv[1]
    match = re.fullmatch(r"v(?P<version>\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?)", tag)
    if match is None:
        print(f"invalid release tag: {tag}", file=sys.stderr)
        return 1
    with Path("pyproject.toml").open("rb") as project_file:
        project = tomllib.load(project_file)
    version = project["project"]["version"]
    if version != match.group("version"):
        print(f"tag {tag} does not match project version {version}", file=sys.stderr)
        return 1
    init_text = Path("src/docchrono/__init__.py").read_text(encoding="utf-8")
    init_match = re.search(r'^__version__\s*=\s*"(?P<version>[^"]+)"$', init_text, re.MULTILINE)
    if init_match is None or init_match.group("version") != version:
        init_version = init_match.group("version") if init_match is not None else "<missing>"
        print(
            f"project version {version} does not match docchrono.__version__ {init_version}",
            file=sys.stderr,
        )
        return 1
    print(f"release identity verified: docchrono {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
