import sys
import pytest
from typer.testing import CliRunner


# Clear out any potential lingering imports from other tests to keep isolation pure
@pytest.fixture(autouse=True)
def cleanup_sys_modules():
    """Ensures TensorFlow isn't pre-cached in memory before running an isolation test."""
    yield
    # Drop cached at_gan references if necessary to force re-evaluation across tests
    for module in list(sys.modules.keys()):
        if module.startswith("at_gan") and "cli" not in module:
            del sys.modules[module]


def test_import_package_is_lazy():
    """Asserts that importing the package namespace does not leak heavy deep learning frameworks."""
    # Ensure no pre-existing leaks
    assert not any("tensorflow" in m for m in sys.modules), "TensorFlow pre-loaded by environment!"

    import at_gan

    assert at_gan is not None
    assert at_gan.__version__ != "0.0.0"

    # Assert TensorFlow stayed completely asleep
    tf_leaks = [m for m in sys.modules if "tensorflow" in m]
    assert len(tf_leaks) == 0, f"Eager import leak detected during bare package import: {tf_leaks}"


def test_lazy_attribute_resolution():
    """Asserts that calling the package shortcuts successfully resolves via __getattr__."""
    import at_gan

    # Trigger the lazy loading hook explicitly
    assert callable(at_gan.train)

    # Verify the backend framework safely initialized on-demand
    assert any("tensorflow" in m for m in sys.modules), "The lazy loader failed to invoke TensorFlow on call."


def test_cli_help_is_fast_and_isolated():
    """Asserts the CLI help menu executes successfully without loading the deep learning backend."""
    # Import the app inside the test execution context to track imports freshly
    from at_gan.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "train" in result.output

    # The ultimate constraint check: --help must never load TensorFlow
    tf_modules = [m for m in sys.modules if "tensorflow" in m]
    assert len(tf_modules) == 0, f"CLI help broke lazy isolation! Loaded modules: {tf_modules}"