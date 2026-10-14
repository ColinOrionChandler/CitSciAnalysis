"""Validate the actual wheel/sdist contents before installation or publication."""

from __future__ import annotations

import argparse
import re
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath


def validate_members(members: dict[str, bytes], *, wheel: bool, expected_version: str | None) -> str:
    """Reject unexpected payloads and validate metadata in one distribution."""
    for name in members:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe archive path: {name}")
        if any(
            part in {"results", "historic_code", ".git", ".env", "_results", "_html", "_build"}
            for part in path.parts
        ):
            raise ValueError(f"Excluded payload: {name}")
        if path.suffix.lower() in {".csv", ".parquet", ".fits", ".fz", ".sqlite", ".db", ".pkl"}:
            raise ValueError(f"Research-data payload: {name}")
        if wheel and path.parts[0] != "citsci_analysis" and not path.parts[0].endswith(".dist-info"):
            raise ValueError(f"Unexpected wheel payload: {name}")
    metadata_paths = [
        name for name in members if (name.endswith(".dist-info/METADATA") if wheel else name == "PKG-INFO")
    ]
    if len(metadata_paths) != 1:
        raise ValueError("Expected exactly one distribution metadata file")
    metadata = BytesParser().parsebytes(members[metadata_paths[0]])
    if metadata["Name"] != "citsci-analysis" or metadata["Requires-Python"] != ">=3.11":
        raise ValueError("Incorrect distribution identity or Python requirement")
    if metadata["License-Expression"] != "BSD-3-Clause":
        raise ValueError("Missing BSD-3-Clause license metadata")
    version = str(metadata["Version"] or "")
    if not version or (expected_version is not None and version != expected_version):
        raise ValueError(f"Unexpected artifact version: {version!r}; expected {expected_version!r}")
    prefix = "citsci_analysis" if wheel else "src/citsci_analysis"
    for filename in ("py.typed", "_version.py", "__init__.py", "__main__.py", "consolidate.py"):
        if f"{prefix}/{filename}" not in members:
            raise ValueError(f"Missing package file: {filename}")
    license_paths = [name for name in members if name.endswith("/licenses/LICENSE")] if wheel else ["LICENSE"]
    if len(license_paths) != 1 or license_paths[0] not in members:
        raise ValueError("Missing license file")
    license_text = members[license_paths[0]].decode()
    for text in ("BSD 3-Clause", "Will Burris", "Colin Chandler", "Chad Trujillo"):
        if text not in license_text:
            raise ValueError(f"Missing license attribution: {text}")
    return version


def validate_distributions(directory: Path, release_tag: str | None = None) -> str:
    """Require one wheel and sdist with matching versions and approved contents."""
    if (
        release_tag is not None
        and re.fullmatch(r"v(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)", release_tag) is None
    ):
        raise ValueError("Expected a canonical vMAJOR.MINOR.PATCH release tag")
    wheels = list(directory.glob("*.whl"))
    sdists = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Expected exactly one wheel and one source distribution")
    expected = release_tag[1:] if release_tag else None
    with zipfile.ZipFile(wheels[0]) as archive:
        wheel_members = {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}
    version = validate_members(wheel_members, wheel=True, expected_version=expected)
    with tarfile.open(sdists[0]) as archive:
        source_members = {}
        roots = set()
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            roots.add(path.parts[0])
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsafe source archive member: {member.name}")
            if member.isfile():
                stream = archive.extractfile(member)
                assert stream is not None
                source_members[str(PurePosixPath(*path.parts[1:]))] = stream.read()
        if len(roots) != 1:
            raise ValueError("Expected a single source distribution root")
    validate_members(source_members, wheel=False, expected_version=version)
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--release-tag")
    args = parser.parse_args()
    print(f"Validated wheel and sdist version {validate_distributions(args.directory, args.release_tag)}")
