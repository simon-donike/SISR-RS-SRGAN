import importlib
import pkgutil

import pytest

# Skip the whole file if torch isn't available (these modules depend on it)
pytest.importorskip("torch")

def test_package_discovery():
    """Ensure packages listed in pyproject are discoverable."""
    packages = {name for _, name, _ in pkgutil.walk_packages(["."], onerror=lambda *_: None)}
    expected = {"opensr_srgan", "utils"}
    assert expected.issubset(packages)
