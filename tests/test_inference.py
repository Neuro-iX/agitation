"""Test the inference.py module."""

import pytest
from monai.data.dataset import Dataset
from pandas.api.types import is_float_dtype
from torch.utils.data import DataLoader

from agitation.data_manager import download_model
from agitation.inference import (
    estimate_motion_dl,
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
