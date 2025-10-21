import importlib
import pkgutil
from pathlib import Path

import pytest

# Skip the whole file if torch isn't available (these modules depend on it)
pytest.importorskip("torch")


def test_package_discovery():
    """Ensure the core package and key subpackages are discoverable."""

    root = Path(__file__).resolve().parents[1]
    discovered = {
        name for _, name, _ in pkgutil.walk_packages([str(root)], prefix="opensr_srgan.")
    }

    assert importlib.import_module("opensr_srgan")
    expected = {"opensr_srgan.model", "opensr_srgan.utils", "opensr_srgan.data"}
    assert expected.issubset(discovered)
