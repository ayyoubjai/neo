import os
import pytest


@pytest.fixture(autouse=True)
def force_example_settings(monkeypatch):
    """Ensure tests use the example settings file instead of a local override.

    This sets the AGI_SETTINGS_PATH environment variable to point at
    `config/settings.example.json` for the duration of tests, preventing
    local `config/settings.local.json` from changing test behavior.
    """
    repo_root = os.path.dirname(os.path.dirname(__file__))
    example = os.path.join(repo_root, "config", "settings.example.json")
    monkeypatch.setenv("AGI_SETTINGS_PATH", example)
    yield