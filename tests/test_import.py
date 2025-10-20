import importlib
import pkgutil

import pytest

pytest.importorskip("torch")


def test_top_level_imports():
    """Ensure key modules can be imported without side effects."""
    modules = [
        "inference",
        "train",
        "opensr_srgan.model",
        "utils.logging_helpers",
        "utils.spectral_helpers",
    ]

    for module_name in modules:
        module = importlib.import_module(module_name)
        assert module is not None


def test_package_discovery():
    """Ensure packages listed in pyproject are discoverable."""
    packages = {name for _, name, _ in pkgutil.walk_packages(["."])}
    expected = {"opensr_srgan", "utils"}
    assert expected.issubset(packages)
