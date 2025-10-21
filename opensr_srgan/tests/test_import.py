import importlib
import pkgutil
import sys
from pathlib import Path

import pytest

# Skip the whole file if torch isn't available (these modules depend on it)
pytest.importorskip("torch")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_package_discovery():
    """Ensure the core package and key subpackages are discoverable."""

    root = Path(__file__).resolve().parents[1]
    discovered = {
        name for _, name, _ in pkgutil.walk_packages([str(root)], prefix="opensr_srgan.")
    }

    assert importlib.import_module("opensr_srgan")
    expected = {"opensr_srgan.model", "opensr_srgan.utils", "opensr_srgan.data"}
    assert expected.issubset(discovered)


def test_import_all_submodules():
    """Import every package module to ensure the reorganised layout works."""

    root = Path(__file__).resolve().parents[1]
    module_names = {
        name
        for _, name, _ in pkgutil.walk_packages([str(root)], prefix="opensr_srgan.")
        if ".tests" not in name
    }

    optional_dependencies = {
        "opensr_srgan.train": {"wandb", "pytorch_lightning"},
    }

    optional_failures = []

    for module_name in sorted(module_names):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing = exc.name or ""
            expected_missing = missing in optional_dependencies.get(module_name, set())
            if expected_missing:
                optional_failures.append((module_name, missing))
                continue
            raise

    if optional_failures:
        missing_info = ", ".join(
            f"{module} (missing {dependency})" for module, dependency in optional_failures
        )
        pytest.skip(f"Optional dependencies missing for modules: {missing_info}")
