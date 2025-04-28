"""Test function from `data_manager`."""

import os

import pytest
import torch

from agitation import config
from agitation.data_manager import MD5MismatchError, download_model, remove_data


def test_download_model_creates_file():
    """Test that `download_model` do create a file."""
    assert not os.path.exists(config.MODEL_PATH)

    download_model()

    assert os.path.exists(config.MODEL_PATH)


def test_download_model_fail(monkeypatch):
    """Test that `download_model` raise a correct error on MD5 mismatch."""
    assert not os.path.exists(config.MODEL_PATH)
    monkeypatch.setattr(
        config,
        "ZENODO_URL",
        "https://zenodo.org/records/15288225/files/SFCN_conv1_noaug_noshift.ckpt?download=1",
    )

    with pytest.raises(MD5MismatchError):
        download_model()


def test_downloaded_model_can_be_loaded():
    """Test that `download_model` file can be loaded."""
    download_model()

    model = torch.jit.load(config.MODEL_PATH)

    assert isinstance(model, torch.jit.ScriptModule)


def test_remove_data_deletes_model():
    """Test `remove_data`."""
    download_model()
    assert os.path.exists(config.MODEL_PATH)

    remove_data()

    assert not os.path.exists(config.MODEL_PATH)
