"""Negative checks for publication guards using synthetic archive members."""

from pathlib import Path

import pytest

from tools.validate_distributions import validate_members


def wheel_members() -> dict[str, bytes]:
    metadata = (
        b"Name: citsci-analysis\nVersion: 0.1.0\nRequires-Python: >=3.11\nLicense-Expression: BSD-3-Clause\n"
    )
    return {
        "citsci_analysis-0.1.0.dist-info/METADATA": metadata,
        "citsci_analysis-0.1.0.dist-info/licenses/LICENSE": Path("LICENSE").read_bytes(),
        **{
            f"citsci_analysis/{name}": b""
            for name in ("__init__.py", "__main__.py", "_version.py", "py.typed", "consolidate.py")
        },
    }


def test_valid_metadata_and_payload() -> None:
    assert validate_members(wheel_members(), wheel=True, expected_version="0.1.0") == "0.1.0"


def test_wrong_built_version_rejected() -> None:
    with pytest.raises(ValueError, match="Unexpected artifact version"):
        validate_members(wheel_members(), wheel=True, expected_version="0.2.0")


@pytest.mark.parametrize("name", ["results/private.txt", "citsci_analysis/input.parquet", "../secret"])
def test_research_and_unsafe_payloads_rejected(name: str) -> None:
    members = wheel_members()
    members[name] = b"synthetic"
    with pytest.raises(ValueError):
        validate_members(members, wheel=True, expected_version="0.1.0")


@pytest.mark.parametrize(
    "name", ["citsci_analysis/py.typed", "citsci_analysis-0.1.0.dist-info/licenses/LICENSE"]
)
def test_missing_required_files_rejected(name: str) -> None:
    members = wheel_members()
    del members[name]
    with pytest.raises(ValueError, match="Missing"):
        validate_members(members, wheel=True, expected_version="0.1.0")
