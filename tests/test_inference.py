"""Test the inference.py module."""

import pytest
from monai.data.dataset import Dataset
from pandas.api.types import is_float_dtype
from torch.utils.data import DataLoader

from agitation.data_manager import download_model
from agitation.inference import (
    bids_to_list,
    clinica_to_list,
    detect_anat,
    detect_t1_linear,
    estimate_motion_dl,
    get_ses,
    get_sub,
    has_session,
)
from agitation.processing import LoadVolume


@pytest.mark.parametrize("cuda", [None, 0])
def test_estimate_motion_dl(cuda):
    """Test `estimate_motion_dl` with cuda or cpu."""
    download_model()
    data = [
        {
            "sub": "sub-000103",
            "ses": "ses-standard",
            "data": "tests/data/clinica/subjects/sub-000103/ses-standard/t1_linear/"
            "sub-000103_ses-standard_acq_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz",
        },
        {
            "sub": "sub-000148",
            "ses": "ses-standard",
            "data": "tests/data/clinica/subjects/sub-000148/ses-standard/t1_linear/"
            "sub-000148_ses-standard_acq_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz",
        },
    ]
    dl = DataLoader(Dataset(data, LoadVolume()), batch_size=2)
    out = estimate_motion_dl(dl, cuda)

    assert len(out) == 2
    assert "sub" in out.columns and out["sub"][0] == "sub-000103"
    assert is_float_dtype(out["motion"])


def test_get_sub():
    """Test `get_sub`."""
    sub_string = (
        "tests/data/clinica/subjects/sub-000103/ses-headmotion2/t1_linear/"
        "sub-000103_ses-headmotion2_acq_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz"
    )
    no_sub = (
        "tests/data/clinica/subjects/b-000103/ses-headmotion2/t1_linear/"
        "s_ses-headmotion2_acq_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz"
    )

    assert get_sub(sub_string) == "sub-000103"
    assert get_sub(no_sub) is None


def test_get_ses():
    """Test `get_ses`."""
    ses_string = (
        "tests/data/clinica/subjects/sub-000103/ses-headmotion2/t1_linear/"
        "sub-000103_ses-headmotion2_acq_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz"
    )
    no_ses = (
        "tests/data/clinica/subjects/sub-000103/c-s/t1_linear/"
        "sub-000103_s-headmotion2_acq_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz"
    )

    assert get_ses(ses_string) == "ses-headmotion2"
    assert get_ses(no_ses) is None


def test_has_session():
    """Test `has_session`."""
    assert has_session("tests/data/bids_sub_ses")
    assert not has_session("tests/data/bids_sub")


def test_bids_to_list_sub():
    """Test `bids_to_list` on sub only dataset."""
    path_to_bids = "tests/data/bids_sub"
    vol_list = bids_to_list(path_to_bids)

    assert len(vol_list) == 2
    assert vol_list[0] == {
        "data": "tests/data/bids_sub/sub-000103/anat/"
        "sub-000103_acq-standard_T1w.nii.gz",
        "sub": "sub-000103",
    }


def test_bids_to_list_sub_ses():
    """Test `bids_to_list` on full bids dataset."""
    path_to_bids = "tests/data/bids_sub_ses"
    vol_list = bids_to_list(path_to_bids)

    assert len(vol_list) == 3
    assert vol_list[0] == {
        "data": "tests/data/bids_sub_ses/sub-000103/ses-headmotion2/anat/"
        "sub-000103_ses-headmotion2_T1w.nii.gz",
        "sub": "sub-000103",
        "ses": "ses-headmotion2",
    }


def test_clinica_to_list():
    """Test `clinica_to_list`."""
    path_to_clinica = "tests/data/clinica"
    vol_list = clinica_to_list(path_to_clinica)

    assert len(vol_list) == 3
    assert vol_list[0] == {
        "data": "tests/data/clinica/subjects/sub-000103/ses-headmotion2/t1_linear/"
        "sub-000103_ses-headmotion2_acq_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz",
        "sub": "sub-000103",
        "ses": "ses-headmotion2",
    }


def test_detect_t1_linear():
    """Test `detect_t1_linear`."""
    assert detect_t1_linear("tests/data/clinica")
    assert not detect_t1_linear("tests/data/bids_sub_ses")
    assert not detect_t1_linear("tests/data/clinicano_t1_linear")


def test_detect_anat():
    """Test `detect_anat`."""
    assert not detect_anat("tests/data/clinica")
    assert detect_anat("tests/data/bids_sub_ses")
    assert detect_anat("tests/data/bids_sub")
    assert not detect_anat("tests/data/bids_sub_ses_noanat")
    assert not detect_anat("tests/data/clinicano_t1_linear")
