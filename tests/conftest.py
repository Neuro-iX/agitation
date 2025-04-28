"""Configuration for tests."""

import os
import shutil

import pytest

from agitation import config

TEST_DATA_DIR = "./tests/tmp_data_dir/"
TEST_MODEL_PATH = "./tests/tmp_data_dir/best_model.pt"


@pytest.fixture(autouse=True, scope="function")
def setup_env(monkeypatch):
    """Change config variables for tests."""
    # Patch config before each test
    monkeypatch.setattr(config, "MODEL_PATH", TEST_MODEL_PATH)
    monkeypatch.setattr(config, "DATA_DIR", TEST_DATA_DIR)


@pytest.fixture(autouse=True, scope="function")
def rm_test_data():
    """Remove any temporary test data."""
    # Cleanup before test
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)
    yield
    # Cleanup after test
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)
